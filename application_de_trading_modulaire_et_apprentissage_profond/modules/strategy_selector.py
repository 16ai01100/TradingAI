import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class StrategySelector:
    """
    Module de sélection automatique de stratégies de trading basé sur les performances historiques
    et les prédictions des modèles d'apprentissage profond.
    """
    
    def __init__(self, backtesting_engineer=None, ml_developer=None, strategy_executor=None, 
                 selection_metric='sharpe_ratio', lookback_period=30, update_frequency=1):
        """
        Initialise le sélecteur de stratégies.
        
        Args:
            backtesting_engineer: Instance de BacktestingEngineer pour évaluer les performances historiques
            ml_developer: Instance de MLDeveloper pour les prédictions de marché
            strategy_executor: Instance de StrategyExecutor pour exécuter les stratégies
            selection_metric (str): Métrique principale pour la sélection ('sharpe_ratio', 'total_return', etc.)
            lookback_period (int): Période de lookback en jours pour l'évaluation des performances
            update_frequency (int): Fréquence de mise à jour de la sélection en jours
        """
        self.backtesting_engineer = backtesting_engineer
        self.ml_developer = ml_developer
        self.strategy_executor = strategy_executor
        self.selection_metric = selection_metric
        self.lookback_period = lookback_period
        self.update_frequency = update_frequency
        
        self.strategy_performances = {}
        self.active_strategy = {}
        self.strategy_history = {}
        self.last_update = {}
        
        self.results_dir = os.path.join(os.getcwd(), 'results')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'strategy_selector.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StrategySelector')
        self.logger.info("Module de sélection de stratégies initialisé")
    
    def evaluate_all_strategies(self, crypto_pair, data, available_strategies=None, parameters=None):
        """
        Évalue toutes les stratégies disponibles sur les données historiques.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            data (pd.DataFrame): Données historiques
            available_strategies (dict): Dictionnaire des stratégies disponibles {nom: fonction}
            parameters (dict): Paramètres pour chaque stratégie
            
        Returns:
            dict: Performances de chaque stratégie
        """
        self.logger.info(f"Évaluation de toutes les stratégies pour {crypto_pair}")
        
        if available_strategies is None and self.strategy_executor:
            available_strategies = self.strategy_executor.strategies
        
        if available_strategies is None:
            self.logger.error("Aucune stratégie disponible pour l'évaluation")
            return {}
        
        parameters = parameters or {}
        performances = {}
        
        # Évaluer chaque stratégie
        for strategy_name, strategy_func in available_strategies.items():
            self.logger.info(f"Évaluation de la stratégie {strategy_name}")
            
            # Paramètres spécifiques à la stratégie ou paramètres par défaut
            strategy_params = parameters.get(strategy_name, {})
            
            # Exécuter le backtest si BacktestingEngineer est disponible
            if self.backtesting_engineer:
                # Stocker temporairement les données
                if crypto_pair not in self.backtesting_engineer.data:
                    self.backtesting_engineer.data[crypto_pair] = data
                
                # Exécuter le backtest
                result = self.backtesting_engineer.run_backtest(
                    strategy_func, crypto_pair, strategy_params
                )
                
                if result:
                    performances[strategy_name] = result
            else:
                # Exécution simplifiée si BacktestingEngineer n'est pas disponible
                signals = strategy_func(data, strategy_params)
                data['signal'] = signals
                data['position'] = data['signal'].fillna(0)
                data['returns'] = data['close'].pct_change()
                data['strategy_returns'] = data['position'].shift(1) * data['returns']
                
                # Calcul des métriques de base
                total_return = (1 + data['strategy_returns']).cumprod().iloc[-1] - 1
                sharpe_ratio = np.sqrt(252) * data['strategy_returns'].mean() / data['strategy_returns'].std()
                max_drawdown = 1 - (1 + data['strategy_returns']).cumprod() / (1 + data['strategy_returns']).cumprod().cummax()
                max_drawdown = max_drawdown.max()
                
                performances[strategy_name] = {
                    'crypto_pair': crypto_pair,
                    'strategy': strategy_name,
                    'parameters': strategy_params,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                }
        
        # Stocker les performances
        self.strategy_performances[crypto_pair] = performances
        
        # Sauvegarder les résultats
        self._save_evaluation_results(crypto_pair, performances)
        
        return performances
    
    def select_best_strategy(self, crypto_pair, market_condition=None, custom_weights=None):
        """
        Sélectionne la meilleure stratégie en fonction des performances historiques et des conditions de marché.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            market_condition (str): Condition de marché actuelle ('bullish', 'bearish', 'sideways', etc.)
            custom_weights (dict): Poids personnalisés pour les différentes métriques
            
        Returns:
            tuple: (nom_stratégie, paramètres)
        """
        self.logger.info(f"Sélection de la meilleure stratégie pour {crypto_pair}")
        
        if crypto_pair not in self.strategy_performances or not self.strategy_performances[crypto_pair]:
            self.logger.error(f"Aucune performance disponible pour {crypto_pair}")
            return None, None
        
        performances = self.strategy_performances[crypto_pair]
        
        # Poids par défaut pour les différentes métriques
        weights = {
            'sharpe_ratio': 0.4,
            'total_return': 0.3,
            'max_drawdown': 0.3  # Négatif car on veut minimiser le drawdown
        }
        
        # Appliquer des poids personnalisés si fournis
        if custom_weights:
            weights.update(custom_weights)
        
        # Ajuster les poids en fonction des conditions de marché
        if market_condition == 'bullish':
            weights['total_return'] = 0.5
            weights['sharpe_ratio'] = 0.3
            weights['max_drawdown'] = 0.2
        elif market_condition == 'bearish':
            weights['max_drawdown'] = 0.5
            weights['sharpe_ratio'] = 0.3
            weights['total_return'] = 0.2
        elif market_condition == 'volatile':
            weights['sharpe_ratio'] = 0.6
            weights['max_drawdown'] = 0.3
            weights['total_return'] = 0.1
        
        # Calculer un score pondéré pour chaque stratégie
        scores = {}
        for strategy_name, result in performances.items():
            # Normaliser les métriques
            sharpe_ratio = result.get('sharpe_ratio', 0)
            total_return = result.get('total_return', 0)
            max_drawdown = result.get('max_drawdown', 1)  # Valeur par défaut élevée
            
            # Calculer le score (inverser le drawdown car on veut le minimiser)
            score = (weights['sharpe_ratio'] * sharpe_ratio + 
                     weights['total_return'] * total_return - 
                     weights['max_drawdown'] * max_drawdown)
            
            scores[strategy_name] = score
        
        # Trouver la stratégie avec le meilleur score
        if not scores:
            return None, None
        
        best_strategy = max(scores.items(), key=lambda x: x[1])[0]
        best_params = performances[best_strategy].get('parameters', {})
        
        # Mettre à jour la stratégie active
        self.active_strategy[crypto_pair] = {
            'strategy_name': best_strategy,
            'parameters': best_params,
            'score': scores[best_strategy],
            'selected_at': datetime.now(),
            'market_condition': market_condition
        }
        
        # Mettre à jour l'historique des stratégies
        if crypto_pair not in self.strategy_history:
            self.strategy_history[crypto_pair] = []
        
        self.strategy_history[crypto_pair].append(self.active_strategy[crypto_pair])
        self.last_update[crypto_pair] = datetime.now()
        
        self.logger.info(f"Meilleure stratégie pour {crypto_pair}: {best_strategy} avec un score de {scores[best_strategy]:.4f}")
        
        return best_strategy, best_params
    
    def detect_market_condition(self, data, window=20):
        """
        Détecte la condition actuelle du marché.
        
        Args:
            data (pd.DataFrame): Données de marché récentes
            window (int): Fenêtre pour le calcul des indicateurs
            
        Returns:
            str: Condition de marché ('bullish', 'bearish', 'sideways', 'volatile')
        """
        self.logger.info("Détection de la condition de marché")
        
        # Calculer les rendements
        returns = data['close'].pct_change().dropna()
        
        # Calculer la tendance (moyenne mobile des rendements)
        trend = returns.rolling(window=window).mean().iloc[-1]
        
        # Calculer la volatilité
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        # Seuils pour la classification
        trend_threshold = 0.001  # 0.1% par jour
        volatility_threshold = 0.02  # 2% d'écart-type
        
        # Classifier la condition de marché
        if volatility > volatility_threshold:
            condition = 'volatile'
        elif trend > trend_threshold:
            condition = 'bullish'
        elif trend < -trend_threshold:
            condition = 'bearish'
        else:
            condition = 'sideways'
        
        self.logger.info(f"Condition de marché détectée: {condition} (tendance: {trend:.4f}, volatilité: {volatility:.4f})")
        
        return condition
    
    def integrate_ml_predictions(self, crypto_pair, data, prediction_weight=0.3):
        """
        Intègre les prédictions des modèles ML dans la sélection de stratégie.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            data (pd.DataFrame): Données de marché récentes
            prediction_weight (float): Poids des prédictions ML dans la décision finale
            
        Returns:
            dict: Scores ajustés des stratégies
        """
        self.logger.info(f"Intégration des prédictions ML pour {crypto_pair}")
        
        if not self.ml_developer or crypto_pair not in self.strategy_performances:
            return self.strategy_performances.get(crypto_pair, {})
        
        # Obtenir la prédiction du modèle ML
        try:
            prediction = self.ml_developer.predict(data, crypto_pair, model_type='lstm')
            if prediction is None:
                return self.strategy_performances.get(crypto_pair, {})
            
            # Calculer la variation prédite
            current_price = data['close'].iloc[-1]
            predicted_change = (prediction - current_price) / current_price
            
            # Ajuster les scores des stratégies en fonction de la prédiction
            adjusted_performances = {}
            for strategy_name, performance in self.strategy_performances[crypto_pair].items():
                # Copier les performances originales
                adjusted_performance = performance.copy()
                
                # Ajuster le rendement total en fonction de la prédiction
                if predicted_change > 0.01:  # Hausse prédite
                    # Favoriser les stratégies avec un bon rendement en marché haussier
                    adjusted_performance['total_return'] = (
                        (1 - prediction_weight) * performance['total_return'] + 
                        prediction_weight * performance['total_return'] * (1 + predicted_change * 10)
                    )
                elif predicted_change < -0.01:  # Baisse prédite
                    # Favoriser les stratégies avec un faible drawdown en marché baissier
                    adjusted_performance['max_drawdown'] = (
                        (1 - prediction_weight) * performance['max_drawdown'] + 
                        prediction_weight * performance['max_drawdown'] * (1 - predicted_change * 10)
                    )
                
                adjusted_performances[strategy_name] = adjusted_performance
            
            self.logger.info(f"Performances ajustées avec les prédictions ML pour {crypto_pair}")
            return adjusted_performances
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration des prédictions ML: {e}")
            return self.strategy_performances.get(crypto_pair, {})
    
    def should_update_strategy(self, crypto_pair):
        """
        Détermine si la stratégie doit être mise à jour.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            bool: True si la stratégie doit être mise à jour, False sinon
        """
        if crypto_pair not in self.last_update:
            return True
        
        time_since_update = datetime.now() - self.last_update[crypto_pair]
        return time_since_update.days >= self.update_frequency
    
    def update_strategy(self, crypto_pair, data):
        """
        Met à jour la stratégie active si nécessaire.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            data (pd.DataFrame): Données de marché récentes
            
        Returns:
            tuple: (nom_stratégie, paramètres)
        """
        self.logger.info(f"Vérification de la mise à jour de stratégie pour {crypto_pair}")
        
        if not self.should_update_strategy(crypto_pair):
            if crypto_pair in self.active_strategy:
                strategy_info = self.active_strategy[crypto_pair]
                self.logger.info(f"Stratégie actuelle maintenue pour {crypto_pair}: {strategy_info['strategy_name']}")
                return strategy_info['strategy_name'], strategy_info['parameters']
            else:
                self.logger.info(f"Aucune stratégie active pour {crypto_pair}, sélection d'une nouvelle stratégie")
        
        # Détecter la condition de marché
        market_condition = self.detect_market_condition(data)
        
        # Évaluer toutes les stratégies disponibles
        if self.strategy_executor:
            self.evaluate_all_strategies(crypto_pair, data, self.strategy_executor.strategies)
        
        # Intégrer les prédictions ML si disponibles
        if self.ml_developer:
            adjusted_performances = self.integrate_ml_predictions(crypto_pair, data)
            if adjusted_performances:
                self.strategy_performances[crypto_pair] = adjusted_performances
        
        # Sélectionner la meilleure stratégie
        best_strategy, best_params = self.select_best_strategy(crypto_pair, market_condition)
        
        # Activer la stratégie si StrategyExecutor est disponible
        if best_strategy and self.strategy_executor:
            self.strategy_executor.activate_strategy(best_strategy, crypto_pair, best_params)
            self.logger.info(f"Stratégie {best_strategy} activée pour {crypto_pair}")
        
        return best_strategy, best_params
    
    def _save_evaluation_results(self, crypto_pair, performances):
        """
        Sauvegarde les résultats d'évaluation des stratégies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            performances (dict): Performances des stratégies
        """
        results_file = os.path.join(self.results_dir, f"{crypto_pair}_strategy_evaluation.json")
        
        # Convertir les résultats en format JSON-compatible
        json_results = {}
        for strategy_name, result in performances.items():
            json_results[strategy_name] = {k: str(v) if isinstance(v, (datetime, timedelta)) else v 
                                         for k, v in result.items()}
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=4)
        
        self.logger.info(f"Résultats d'évaluation sauvegardés dans {results_file}")
    
    def plot_strategy_performance(self, crypto_pair):
        """
        Génère un graphique comparatif des performances des stratégies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            str: Chemin du fichier de graphique généré
        """
        if crypto_pair not in self.strategy_performances or not self.strategy_performances[crypto_pair]:
            self.logger.error(f"Aucune performance disponible pour {crypto_pair}")
            return None
        
        performances = self.strategy_performances[crypto_pair]
        
        # Extraire les métriques pour chaque stratégie
        strategies = list(performances.keys())
        sharpe_ratios = [performances[s].get('sharpe_ratio', 0) for s in strategies]
        total_returns = [performances[s].get('total_return', 0) * 100 for s in strategies]  # En pourcentage
        max_drawdowns = [performances[s].get('max_drawdown', 0) * 100 for s in strategies]  # En pourcentage
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Largeur des barres
        width = 0.25
        
        # Positions des barres
        x = np.arange(len(strategies))
        
        # Tracer les barres
        ax.bar(x - width, sharpe_ratios, width, label='Ratio de Sharpe')
        ax.bar(x, total_returns, width, label='Rendement total (%)')
        ax.bar(x + width, max_drawdowns, width, label='Drawdown max (%)')
        
        # Ajouter les étiquettes et la légende
        ax.set_xlabel('Stratégies')
        ax.set_ylabel('Valeurs')
        ax.set_title(f'Comparaison des performances des stratégies pour {crypto_pair}')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        
        # Ajuster la mise en page
        fig.tight_layout()
        
        # Sauvegarder le graphique
        plot_file = os.path.join(self.results_dir, f"{crypto_pair}_strategy_comparison.png")
        plt.savefig(plot_file)
        plt.close()
        
        self.logger.info(f"Graphique de comparaison des stratégies sauvegardé dans {plot_file}")
        
        return plot_file
    
    def get_strategy_history(self, crypto_pair):
        """
        Récupère l'historique des stratégies sélectionnées pour une paire de crypto-monnaies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            list: Historique des stratégies
        """
        return self.strategy_history.get(crypto_pair, [])
    
    def get_active_strategy(self, crypto_pair):
        """
        Récupère la stratégie active pour une paire de crypto-monnaies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            dict: Informations sur la stratégie active
        """
        return self.active_strategy.get(crypto_pair, None)