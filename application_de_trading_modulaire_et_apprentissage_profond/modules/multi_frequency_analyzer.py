import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class MultiFrequencyAnalyzer:
    """
    Module d'analyse multi-fréquence qui combine des données de différentes temporalités
    et sources (temps réel et on-chain) pour améliorer la prise de décision.
    """
    
    def __init__(self, data_collector=None, binance_connector=None, timeframes=None, 
                 on_chain_sources=None, correlation_threshold=0.7, max_workers=4):
        """
        Initialise l'analyseur multi-fréquence.
        
        Args:
            data_collector: Instance de DataCollector pour récupérer les données historiques
            binance_connector: Instance de BinanceConnector pour les données temps réel
            timeframes (list): Liste des timeframes à analyser (ex: ['1m', '5m', '15m', '1h', '4h', '1d'])
            on_chain_sources (dict): Sources de données on-chain à intégrer
            correlation_threshold (float): Seuil de corrélation pour la détection de signaux
            max_workers (int): Nombre maximum de workers pour le traitement parallèle
        """
        self.data_collector = data_collector
        self.binance_connector = binance_connector
        self.timeframes = timeframes or ['1m', '5m', '15m', '1h', '4h', '1d']
        self.on_chain_sources = on_chain_sources or {}
        self.correlation_threshold = correlation_threshold
        self.max_workers = max_workers
        
        self.data_cache = {}
        self.on_chain_data = {}
        self.combined_signals = {}
        self.frequency_weights = {}
        self.cross_frequency_patterns = {}
        self.last_update = {}
        
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        self.data_dir = os.path.join(os.getcwd(), 'data')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'multi_frequency_analyzer.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MultiFrequencyAnalyzer')
        self.logger.info("Module d'analyse multi-fréquence initialisé")
        
        # Initialiser les poids des fréquences (par défaut, poids égaux)
        self._initialize_frequency_weights()
    
    def _initialize_frequency_weights(self):
        """
        Initialise les poids pour chaque timeframe.
        Par défaut, les timeframes plus courts ont un poids plus faible
        et les timeframes plus longs ont un poids plus élevé.
        """
        weights = {}
        total_timeframes = len(self.timeframes)
        
        for i, tf in enumerate(self.timeframes):
            # Poids croissant avec la durée du timeframe
            weight = (i + 1) / total_timeframes
            weights[tf] = weight
        
        # Normaliser les poids pour qu'ils somment à 1
        total_weight = sum(weights.values())
        for tf in weights:
            weights[tf] /= total_weight
        
        self.frequency_weights = weights
        self.logger.info(f"Poids des fréquences initialisés: {self.frequency_weights}")
    
    def update_frequency_weights(self, performance_metrics):
        """
        Met à jour les poids des fréquences en fonction des performances passées.
        
        Args:
            performance_metrics (dict): Métriques de performance par timeframe
        """
        if not performance_metrics:
            return
        
        # Extraire les métriques pertinentes (ex: précision des signaux)
        accuracy = {}
        for tf, metrics in performance_metrics.items():
            if tf in self.timeframes:
                accuracy[tf] = metrics.get('accuracy', 0.5)  # Valeur par défaut: 0.5
        
        # Mettre à jour les poids en fonction de la précision
        if accuracy:
            total_accuracy = sum(accuracy.values())
            if total_accuracy > 0:
                for tf in self.timeframes:
                    if tf in accuracy:
                        self.frequency_weights[tf] = accuracy[tf] / total_accuracy
                    else:
                        self.frequency_weights[tf] = 0.1  # Valeur par défaut pour les timeframes sans métrique
                
                # Normaliser les poids
                total_weight = sum(self.frequency_weights.values())
                for tf in self.frequency_weights:
                    self.frequency_weights[tf] /= total_weight
                
                self.logger.info(f"Poids des fréquences mis à jour: {self.frequency_weights}")
    
    def fetch_multi_timeframe_data(self, crypto_pair, lookback_periods=None):
        """
        Récupère les données pour plusieurs timeframes en parallèle.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            lookback_periods (dict): Nombre de périodes à récupérer par timeframe
            
        Returns:
            dict: Données par timeframe
        """
        if not self.data_collector:
            self.logger.error("DataCollector non disponible pour récupérer les données")
            return {}
        
        # Périodes de lookback par défaut
        if lookback_periods is None:
            lookback_periods = {
                '1m': 1440,    # 1 jour
                '5m': 1440,    # 5 jours
                '15m': 960,    # 10 jours
                '1h': 720,     # 30 jours
                '4h': 360,     # 60 jours
                '1d': 365      # 1 an
            }
        
        # Fonction pour récupérer les données d'un timeframe spécifique
        def fetch_timeframe(tf):
            periods = lookback_periods.get(tf, 100)  # Valeur par défaut: 100 périodes
            end_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculer la date de début en fonction du timeframe et du nombre de périodes
            if tf.endswith('m'):
                minutes = int(tf[:-1]) * periods
                start_str = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
            elif tf.endswith('h'):
                hours = int(tf[:-1]) * periods
                start_str = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            elif tf.endswith('d'):
                days = int(tf[:-1]) * periods
                start_str = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
            else:
                self.logger.warning(f"Format de timeframe non reconnu: {tf}")
                return tf, None
            
            try:
                data = self.data_collector.get_historical_klines(crypto_pair, tf, start_str, end_str)
                return tf, data
            except Exception as e:
                self.logger.error(f"Erreur lors de la récupération des données pour {crypto_pair} en {tf}: {e}")
                return tf, None
        
        # Récupérer les données en parallèle
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_tf = {executor.submit(fetch_timeframe, tf): tf for tf in self.timeframes}
            for future in as_completed(future_to_tf):
                tf, data = future.result()
                if data is not None:
                    results[tf] = data
        
        # Mettre à jour le cache
        self.data_cache[crypto_pair] = results
        self.last_update[crypto_pair] = datetime.now()
        
        self.logger.info(f"Données multi-timeframes récupérées pour {crypto_pair}: {list(results.keys())}")
        return results
    
    def fetch_on_chain_data(self, crypto_pair, metrics=None):
        """
        Récupère les données on-chain pour une crypto-monnaie.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            metrics (list): Liste des métriques on-chain à récupérer
            
        Returns:
            dict: Données on-chain
        """
        if not self.on_chain_sources:
            self.logger.warning("Aucune source de données on-chain configurée")
            return {}
        
        # Extraire le symbole de base (ex: BTC de BTCUSDT)
        base_symbol = crypto_pair.split('USDT')[0] if 'USDT' in crypto_pair else crypto_pair
        
        # Métriques par défaut
        if metrics is None:
            metrics = [
                'active_addresses',
                'transaction_count',
                'transaction_volume',
                'average_transaction_value',
                'mining_difficulty',
                'hash_rate',
                'exchange_flow'
            ]
        
        on_chain_data = {}
        
        # Simuler la récupération de données on-chain (à remplacer par l'intégration réelle)
        # Dans une implémentation réelle, on utiliserait des API comme Glassnode, CryptoQuant, etc.
        for metric in metrics:
            # Simulation de données
            on_chain_data[metric] = np.random.normal(size=30)  # 30 jours de données simulées
        
        # Stocker les données
        self.on_chain_data[base_symbol] = on_chain_data
        
        self.logger.info(f"Données on-chain récupérées pour {base_symbol}: {list(on_chain_data.keys())}")
        return on_chain_data
    
    def analyze_cross_frequency_patterns(self, crypto_pair, data=None):
        """
        Analyse les patterns à travers différentes fréquences temporelles.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            data (dict): Données multi-timeframes (si None, utilise les données en cache)
            
        Returns:
            dict: Patterns détectés et leur force
        """
        # Utiliser les données en cache si aucune donnée n'est fournie
        if data is None:
            if crypto_pair not in self.data_cache:
                self.logger.warning(f"Aucune donnée en cache pour {crypto_pair}")
                return {}
            data = self.data_cache[crypto_pair]
        
        patterns = {}
        
        # Détecter les divergences entre timeframes
        patterns['divergences'] = self._detect_divergences(data)
        
        # Détecter les confirmations de tendance à travers les timeframes
        patterns['trend_confirmations'] = self._detect_trend_confirmations(data)
        
        # Détecter les niveaux de support/résistance communs
        patterns['support_resistance'] = self._detect_common_support_resistance(data)
        
        # Détecter les patterns de volatilité
        patterns['volatility_patterns'] = self._detect_volatility_patterns(data)
        
        # Stocker les patterns détectés
        self.cross_frequency_patterns[crypto_pair] = patterns
        
        self.logger.info(f"Patterns cross-fréquence analysés pour {crypto_pair}")
        return patterns
    
    def _detect_divergences(self, data):
        """
        Détecte les divergences entre différents timeframes.
        
        Args:
            data (dict): Données multi-timeframes
            
        Returns:
            dict: Divergences détectées
        """
        divergences = {}
        
        # Comparer les tendances entre timeframes adjacents
        sorted_timeframes = sorted(data.keys(), key=self._timeframe_to_minutes)
        
        for i in range(len(sorted_timeframes) - 1):
            tf1 = sorted_timeframes[i]
            tf2 = sorted_timeframes[i + 1]
            
            # Calculer la tendance pour chaque timeframe
            trend1 = self._calculate_trend(data[tf1])
            trend2 = self._calculate_trend(data[tf2])
            
            # Détecter les divergences
            if trend1 * trend2 < 0:  # Tendances opposées
                divergence_strength = abs(trend1 - trend2) / 2
                divergences[f"{tf1}_vs_{tf2}"] = {
                    'type': 'bullish' if trend1 > trend2 else 'bearish',
                    'strength': divergence_strength
                }
        
        return divergences
    
    def _detect_trend_confirmations(self, data):
        """
        Détecte les confirmations de tendance à travers les timeframes.
        
        Args:
            data (dict): Données multi-timeframes
            
        Returns:
            dict: Confirmations de tendance
        """
        confirmations = {}
        
        # Calculer la tendance pour chaque timeframe
        trends = {tf: self._calculate_trend(data[tf]) for tf in data}
        
        # Compter les tendances haussières et baissières
        bullish_count = sum(1 for trend in trends.values() if trend > 0.05)
        bearish_count = sum(1 for trend in trends.values() if trend < -0.05)
        neutral_count = len(trends) - bullish_count - bearish_count
        
        # Calculer la force de la confirmation
        if bullish_count > len(trends) / 2:
            confirmation_type = 'bullish'
            confirmation_strength = bullish_count / len(trends)
        elif bearish_count > len(trends) / 2:
            confirmation_type = 'bearish'
            confirmation_strength = bearish_count / len(trends)
        else:
            confirmation_type = 'neutral'
            confirmation_strength = neutral_count / len(trends)
        
        confirmations['overall'] = {
            'type': confirmation_type,
            'strength': confirmation_strength,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count
        }
        
        return confirmations
    
    def _detect_common_support_resistance(self, data):
        """
        Détecte les niveaux de support/résistance communs à travers les timeframes.
        
        Args:
            data (dict): Données multi-timeframes
            
        Returns:
            dict: Niveaux de support/résistance
        """
        levels = {}
        
        # Extraire les niveaux de prix importants pour chaque timeframe
        price_levels = {}
        for tf, df in data.items():
            if df is None or len(df) < 20:
                continue
                
            # Calculer les niveaux de prix importants (max, min, ouverture, clôture)
            highs = df['high'].nlargest(5).tolist()
            lows = df['low'].nsmallest(5).tolist()
            price_levels[tf] = highs + lows
        
        # Regrouper les niveaux similaires
        all_levels = []
        for tf_levels in price_levels.values():
            all_levels.extend(tf_levels)
        
        # Regrouper les niveaux proches (tolérance de 0.5%)
        grouped_levels = []
        for level in sorted(all_levels):
            if not grouped_levels or abs(level - grouped_levels[-1][0]) / grouped_levels[-1][0] > 0.005:
                grouped_levels.append([level, 1])
            else:
                # Mettre à jour le niveau existant (moyenne pondérée)
                existing_level, count = grouped_levels[-1]
                new_level = (existing_level * count + level) / (count + 1)
                grouped_levels[-1] = [new_level, count + 1]
        
        # Filtrer les niveaux avec au moins 2 occurrences
        significant_levels = [level for level, count in grouped_levels if count >= 2]
        
        # Classer les niveaux comme support ou résistance
        current_price = data[self.timeframes[0]]['close'].iloc[-1] if data[self.timeframes[0]] is not None else 0
        
        supports = [level for level in significant_levels if level < current_price]
        resistances = [level for level in significant_levels if level > current_price]
        
        levels['supports'] = sorted(supports, reverse=True)
        levels['resistances'] = sorted(resistances)
        
        return levels
    
    def _detect_volatility_patterns(self, data):
        """
        Détecte les patterns de volatilité à travers les timeframes.
        
        Args:
            data (dict): Données multi-timeframes
            
        Returns:
            dict: Patterns de volatilité
        """
        volatility_patterns = {}
        
        # Calculer la volatilité pour chaque timeframe
        volatilities = {}
        for tf, df in data.items():
            if df is None or len(df) < 20:
                continue
                
            # Calculer la volatilité (écart-type des rendements)
            returns = df['close'].pct_change().dropna()
            volatility = returns.std()
            volatilities[tf] = volatility
        
        # Comparer la volatilité actuelle à la volatilité historique
        for tf, vol in volatilities.items():
            if df is None or len(df) < 40:
                continue
                
            # Calculer la volatilité historique (période précédente de même longueur)
            half_len = len(df) // 2
            historical_returns = df['close'].iloc[:half_len].pct_change().dropna()
            historical_volatility = historical_returns.std()
            
            # Calculer le ratio de volatilité
            vol_ratio = vol / historical_volatility if historical_volatility > 0 else 1.0
            
            # Classifier le pattern de volatilité
            if vol_ratio > 1.5:
                pattern = 'increasing'
            elif vol_ratio < 0.67:
                pattern = 'decreasing'
            else:
                pattern = 'stable'
            
            volatility_patterns[tf] = {
                'current': vol,
                'historical': historical_volatility,
                'ratio': vol_ratio,
                'pattern': pattern
            }
        
        return volatility_patterns
    
    def _timeframe_to_minutes(self, timeframe):
        """
        Convertit un timeframe en minutes pour faciliter le tri.
        
        Args:
            timeframe (str): Timeframe (ex: '1m', '1h', '1d')
            
        Returns:
            int: Équivalent en minutes
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        else:
            return 0
    
    def _calculate_trend(self, data):
        """
        Calcule la tendance d'une série de données.
        
        Args:
            data (pd.DataFrame): Données de marché
            
        Returns:
            float: Force de la tendance (-1 à 1)
        """
        if data is None or len(data) < 10:
            return 0
        
        # Utiliser une régression linéaire simple sur les prix de clôture
        y = data['close'].values
        x = np.arange(len(y))
        
        # Calculer la pente de la régression linéaire
        slope, _ = np.polyfit(x, y, 1)
        
        # Normaliser la pente par rapport au prix moyen
        normalized_slope = slope / np.mean(y) * 100
        
        # Limiter la valeur entre -1 et 1
        trend = max(min(normalized_slope, 1), -1)
        
        return trend
    
    def generate_combined_signal(self, crypto_pair, strategy_signals=None):
        """
        Génère un signal combiné à partir des signaux de différentes fréquences et stratégies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            strategy_signals (dict): Signaux générés par différentes stratégies
            
        Returns:
            dict: Signal combiné et sa force
        """
        if crypto_pair not in self.cross_frequency_patterns:
            self.logger.warning(f"Aucun pattern cross-fréquence disponible pour {crypto_pair}")
            return {'signal': 0, 'strength': 0, 'confidence': 0}
        
        patterns = self.cross_frequency_patterns[crypto_pair]
        
        # Initialiser les scores
        bullish_score = 0
        bearish_score = 0
        total_weight = 0
        
        # Intégrer les divergences
        if 'divergences' in patterns:
            for div_key, div_info in patterns['divergences'].items():
                weight = 0.3  # Poids des divergences
                if div_info['type'] == 'bullish':
                    bullish_score += weight * div_info['strength']
                else:
                    bearish_score += weight * div_info['strength']
                total_weight += weight
        
        # Intégrer les confirmations de tendance
        if 'trend_confirmations' in patterns and 'overall' in patterns['trend_confirmations']:
            conf_info = patterns['trend_confirmations']['overall']
            weight = 0.4  # Poids des confirmations de tendance
            if conf_info['type'] == 'bullish':
                bullish_score += weight * conf_info['strength']
            elif conf_info['type'] == 'bearish':
                bearish_score += weight * conf_info['strength']
            total_weight += weight
        
        # Intégrer les niveaux de support/résistance
        if 'support_resistance' in patterns:
            sr_levels = patterns['support_resistance']
            current_price = self.data_cache[crypto_pair][self.timeframes[0]]['close'].iloc[-1] if crypto_pair in self.data_cache else 0
            
            # Vérifier la proximité des supports/résistances
            if 'supports' in sr_levels and sr_levels['supports']:
                closest_support = max(sr_levels['supports'])
                support_proximity = 1 - (current_price - closest_support) / current_price if current_price > 0 else 0
                if support_proximity > 0.95:  # Support très proche
                    weight = 0.2
                    bullish_score += weight * support_proximity
                    total_weight += weight
            
            if 'resistances' in sr_levels and sr_levels['resistances']:
                closest_resistance = min(sr_levels['resistances'])
                resistance_proximity = 1 - (closest_resistance - current_price) / current_price if current_price > 0 else 0
                if resistance_proximity > 0.95:  # Résistance très proche
                    weight = 0.2
                    bearish_score += weight * resistance_proximity
                    total_weight += weight
        
        # Intégrer les signaux des stratégies si fournis
        if strategy_signals:
            weight = 0.3  # Poids des signaux de stratégie
            for strategy, signal in strategy_signals.items():
                if signal > 0:
                    bullish_score += weight * abs(signal)
                elif signal < 0:
                    bearish_score += weight * abs(signal)
                total_weight += weight
        
        # Calculer le signal final
        if total_weight > 0:
            bullish_score /= total_weight
            bearish_score /= total_weight
        
        signal_strength = bullish_score - bearish_score
        signal = 1 if signal_strength > 0.2 else (-1 if signal_strength < -0.2 else 0)
        confidence = abs(signal_strength)
        
        # Stocker le signal combiné
        combined_signal = {
            'signal': signal,
            'strength': signal_strength,
            'confidence': confidence,
            'bullish_score': bullish_score,
            'bearish_score': bearish_score,
            'timestamp': datetime.now()
        }
        
        self.combined_signals[crypto_pair] = combined_signal
        
        self.logger.info(f"Signal combiné généré pour {crypto_pair}: {signal} (force: {signal_strength:.4f}, confiance: {confidence:.4f})")
        return combined_signal
    
    def plot_multi_timeframe_analysis(self, crypto_pair, output_file=None):
        """
        Génère un graphique d'analyse multi-timeframe.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            output_file (str): Chemin du fichier de sortie pour le graphique
            
        Returns:
            bool: True si le graphique a été généré avec succès, False sinon
        """
        if crypto_pair not in self.data_cache:
            self.logger.warning(f"Aucune donnée disponible pour {crypto_pair}")
            return False
        
        try:
            # Créer une figure avec plusieurs sous-graphiques
            num_timeframes = len(self.data_cache[crypto_pair])
            fig, axs = plt.subplots(num_timeframes, 1, figsize=(12, 4 * num_timeframes), sharex=False)
            
            # Trier les timeframes
            sorted_timeframes = sorted(self.data_cache[crypto_pair].keys(), key=self._timeframe_to_minutes)
            
            # Tracer les données pour chaque timeframe
            for i, tf in enumerate(sorted_timeframes):
                data = self.data_cache[crypto_pair][tf]
                if data is None or len(data) == 0:
                    continue
                
                ax = axs[i] if num_timeframes > 1 else axs
                
                # Tracer les prix de clôture
                ax.plot(data.index, data['close'], label=f'Prix de clôture ({tf})')
                
                # Ajouter les moyennes mobiles
                if len(data) >= 20:
                    data['sma_20'] = data['close'].rolling(window=20).mean()
                    ax.plot(data.index, data['sma_20'], 'r--', label='SMA 20')
                
                if len(data) >= 50:
                    data['sma_50'] = data['close'].rolling(window=50).mean()
                    ax.plot(data.index, data['sma_50'], 'g--', label='SMA 50')
                
                # Ajouter les niveaux de support/résistance si disponibles
                if crypto_pair in self.cross_frequency_patterns and 'support_resistance' in self.cross_frequency_patterns[crypto_pair]:
                    sr_levels = self.cross_frequency_patterns[crypto_pair]['support_resistance']
                    
                    for support in sr_levels.get('supports', [])[:3]:  # Limiter à 3 niveaux
                        ax.axhline(y=support, color='g', linestyle='-', alpha=0.5)
                    
                    for resistance in sr_levels.get('resistances', [])[:3]:  # Limiter à 3 niveaux
                        ax.axhline(y=resistance, color='r', linestyle='-', alpha=0.5)
                
                # Configurer le graphique
                ax.set_title(f'Analyse {crypto_pair} - Timeframe {tf}')
                ax.set_ylabel('Prix')
                ax.legend()
                ax.grid(True)
            
            # Configurer le graphique du bas
            if num_timeframes > 1:
                axs[-1].set_xlabel('Date')
            else:
                axs.set_xlabel('Date')