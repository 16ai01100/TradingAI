import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException

class StrategyExecutor:
    """
    Module d'exécution de stratégies de trading pour Binance.
    """
    
    def __init__(self, api_key=None, api_secret=None, crypto_pairs=None, strategy_parameters=None):
        """
        Initialise l'exécuteur de stratégies.
        
        Args:
            api_key (str): Clé API Binance
            api_secret (str): Secret API Binance
            crypto_pairs (list): Liste des paires de crypto-monnaies à trader
            strategy_parameters (dict): Paramètres des stratégies de trading
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.crypto_pairs = crypto_pairs or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.strategy_parameters = strategy_parameters or {}
        self.client = None
        self.strategies = {}
        self.active_strategies = {}
        self.orders = {}
        self.data_dir = os.path.join(os.getcwd(), 'data')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'strategy_executor.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('StrategyExecutor')
        
        # Initialisation du client Binance si les clés API sont fournies
        if api_key and api_secret:
            self.connect_to_binance()
        
        # Initialisation des stratégies disponibles
        self._initialize_strategies()
    
    def connect_to_binance(self):
        """
        Établit une connexion à l'API Binance.
        
        Returns:
            bool: True si la connexion est établie avec succès, False sinon
        """
        try:
            self.client = Client(self.api_key, self.api_secret)
            self.logger.info("Connexion à Binance établie avec succès")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la connexion à Binance: {e}")
            return False
    
    def _initialize_strategies(self):
        """
        Initialise les stratégies de trading disponibles.
        """
        self.strategies = {
            'moving_average_crossover': self.moving_average_crossover_strategy,
            'rsi': self.rsi_strategy,
            'bollinger_bands': self.bollinger_bands_strategy,
            'macd': self.macd_strategy,
            'ichimoku': self.ichimoku_strategy
        }
        self.logger.info(f"Stratégies disponibles: {list(self.strategies.keys())}")
    
    def activate_strategy(self, strategy_name, crypto_pair, parameters=None):
        """
        Active une stratégie de trading pour une paire de crypto-monnaies.
        
        Args:
            strategy_name (str): Nom de la stratégie à activer
            crypto_pair (str): Paire de crypto-monnaies
            parameters (dict): Paramètres spécifiques à la stratégie
            
        Returns:
            bool: True si la stratégie est activée avec succès, False sinon
        """
        if strategy_name not in self.strategies:
            self.logger.error(f"Stratégie {strategy_name} non disponible")
            return False
        
        if crypto_pair not in self.crypto_pairs:
            self.logger.error(f"Paire {crypto_pair} non configurée")
            return False
        
        # Fusionner les paramètres spécifiques avec les paramètres globaux
        merged_parameters = {}
        if strategy_name in self.strategy_parameters:
            merged_parameters.update(self.strategy_parameters[strategy_name])
        if parameters:
            merged_parameters.update(parameters)
        
        # Activer la stratégie
        key = f"{strategy_name}_{crypto_pair}"
        self.active_strategies[key] = {
            'strategy_name': strategy_name,
            'crypto_pair': crypto_pair,
            'parameters': merged_parameters,
            'active': True,
            'last_run': None,
            'signals': []
        }
        
        self.logger.info(f"Stratégie {strategy_name} activée pour {crypto_pair} avec paramètres {merged_parameters}")
        return True
    
    def deactivate_strategy(self, strategy_name, crypto_pair):
        """
        Désactive une stratégie de trading pour une paire de crypto-monnaies.
        
        Args:
            strategy_name (str): Nom de la stratégie à désactiver
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            bool: True si la stratégie est désactivée avec succès, False sinon
        """
        key = f"{strategy_name}_{crypto_pair}"
        if key in self.active_strategies:
            self.active_strategies[key]['active'] = False
            self.logger.info(f"Stratégie {strategy_name} désactivée pour {crypto_pair}")
            return True
        else:
            self.logger.error(f"Stratégie {strategy_name} non active pour {crypto_pair}")
            return False
    
    def execute_strategies(self, data_dict):
        """
        Exécute toutes les stratégies actives sur les données fournies.
        
        Args:
            data_dict (dict): Dictionnaire des DataFrames par paire de crypto-monnaies
            
        Returns:
            dict: Signaux générés par les stratégies
        """
        signals = {}
        
        for key, strategy_info in self.active_strategies.items():
            if not strategy_info['active']:
                continue
            
            strategy_name = strategy_info['strategy_name']
            crypto_pair = strategy_info['crypto_pair']
            parameters = strategy_info['parameters']
            
            if crypto_pair not in data_dict:
                self.logger.warning(f"Données non disponibles pour {crypto_pair}")
                continue
            
            data = data_dict[crypto_pair]
            
            # Exécuter la stratégie
            strategy_func = self.strategies[strategy_name]
            signal = strategy_func(data, parameters)
            
            # Enregistrer le signal
            if key not in signals:
                signals[key] = []
            signals[key].append(signal)
            
            # Mettre à jour les informations de la stratégie
            self.active_strategies[key]['last_run'] = datetime.now()
            self.active_strategies[key]['signals'].append(signal)
            
            self.logger.info(f"Stratégie {strategy_name} exécutée pour {crypto_pair}, signal: {signal}")
        
        return signals
    
    def place_order(self, crypto_pair, side, quantity, order_type='MARKET'):
        """
        Place un ordre sur Binance.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            side (str): Côté de l'ordre ('BUY' ou 'SELL')
            quantity (float): Quantité à acheter/vendre
            order_type (str): Type d'ordre ('MARKET' ou 'LIMIT')
            
        Returns:
            dict: Informations sur l'ordre placé
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Placement d'un ordre {side} pour {quantity} {crypto_pair}")
            
            # Placer l'ordre
            if order_type == 'MARKET':
                order = self.client.create_order(
                    symbol=crypto_pair,
                    side=side,
                    type=order_type,
                    quantity=quantity
                )
            else:
                # Pour les ordres LIMIT, il faudrait spécifier un prix
                self.logger.error("Les ordres LIMIT ne sont pas encore implémentés")
                return None
            
            # Enregistrer l'ordre
            if crypto_pair not in self.orders:
                self.orders[crypto_pair] = []
            self.orders[crypto_pair].append(order)
            
            self.logger.info(f"Ordre placé avec succès: {order}")
            return order
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            return None
    
    def get_account_balance(self):
        """
        Récupère le solde du compte.
        
        Returns:
            dict: Solde du compte
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info("Récupération du solde du compte")
            account_info = self.client.get_account()
            balances = account_info['balances']
            
            # Filtrer les soldes non nuls
            non_zero_balances = [b for b in balances if float(b['free']) > 0 or float(b['locked']) > 0]
            
            return non_zero_balances
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération du solde: {e}")
            return None
    
    def calculate_position_size(self, crypto_pair, risk_percentage=1.0):
        """
        Calcule la taille de position en fonction du risque.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            risk_percentage (float): Pourcentage du capital à risquer
            
        Returns:
            float: Taille de la position
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return 0.0
        
        try:
            # Récupérer le solde en USDT
            account_info = self.client.get_account()
            balances = account_info['balances']
            usdt_balance = next((float(b['free']) for b in balances if b['asset'] == 'USDT'), 0.0)
            
            # Calculer le montant à risquer
            risk_amount = usdt_balance * (risk_percentage / 100.0)
            
            # Récupérer le prix actuel
            ticker = self.client.get_symbol_ticker(symbol=crypto_pair)
            current_price = float(ticker['price'])
            
            # Calculer la quantité
            quantity = risk_amount / current_price
            
            # Arrondir à la précision requise
            info = self.client.get_symbol_info(crypto_pair)
            lot_size_filter = next((f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                quantity = round(quantity, precision)
            
            self.logger.info(f"Taille de position calculée pour {crypto_pair}: {quantity}")
            return quantity
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.0
    
    # Implémentation des stratégies de trading
    
    def moving_average_crossover_strategy(self, data, parameters):
        """
        Stratégie de croisement de moyennes mobiles.
        
        Args:
            data (pd.DataFrame): Données de marché
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            int: Signal de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        # Paramètres par défaut
        short_window = parameters.get('short_window', 50)
        long_window = parameters.get('long_window', 200)
        
        # Calcul des moyennes mobiles
        data['short_ma'] = data['close'].rolling(window=short_window).mean()
        data['long_ma'] = data['close'].rolling(window=long_window).mean()
        
        # Génération du signal
        signal = 0
        
        # Vérifier s'il y a suffisamment de données
        if len(data) < long_window:
            self.logger.warning(f"Données insuffisantes pour la stratégie de croisement de moyennes mobiles")
            return signal
        
        # Vérifier le croisement
        if data['short_ma'].iloc[-2] < data['long_ma'].iloc[-2] and data['short_ma'].iloc[-1] > data['long_ma'].iloc[-1]:
            signal = 1  # Signal d'achat
        elif data['short_ma'].iloc[-2] > data['long_ma'].iloc[-2] and data['short_ma'].iloc[-1] < data['long_ma'].iloc[-1]:
            signal = -1  # Signal de vente
        
        return signal
    
    def rsi_strategy(self, data, parameters):
        """
        Stratégie basée sur l'indice de force relative (RSI).
        
        Args:
            data (pd.DataFrame): Données de marché
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            int: Signal de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        # Paramètres par défaut
        period = parameters.get('period', 14)
        oversold = parameters.get('oversold', 30)
        overbought = parameters.get('overbought', 70)
        
        # Calcul du RSI
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Génération du signal
        signal = 0
        
        # Vérifier s'il y a suffisamment de données
        if len(data) < period:
            self.logger.warning(f"Données insuffisantes pour la stratégie RSI")
            return signal
        
        # Vérifier les conditions de surachat/survente
        if data['rsi'].iloc[-1] < oversold:
            signal = 1  # Signal d'achat (survente)
        elif data['rsi'].iloc[-1] > overbought:
            signal = -1  # Signal de vente (surachat)
        
        return signal
    
    def bollinger_bands_strategy(self, data, parameters):
        """
        Stratégie basée sur les bandes de Bollinger.
        
        Args:
            data (pd.DataFrame): Données de marché
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            int: Signal de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        # Paramètres par défaut
        window = parameters.get('window', 20)
        num_std = parameters.get('num_std', 2)
        
        # Calcul des bandes de Bollinger
        data['ma'] = data['close'].rolling(window=window).mean()
        data['std'] = data['close'].rolling(window=window).std()
        data['upper_band'] = data['ma'] + (data['std'] * num_std)
        data['lower_band'] = data['ma'] - (data['std'] * num_std)
        
        # Génération du signal
        signal = 0
        
        # Vérifier s'il y a suffisamment de données
        if len(data) < window:
            self.logger.warning(f"Données insuffisantes pour la stratégie des bandes de Bollinger")
            return signal
        
        # Vérifier les conditions de franchissement des bandes
        if data['close'].iloc[-1] < data['lower_band'].iloc[-1]:
            signal = 1  # Signal d'achat (prix sous la bande inférieure)
        elif data['close'].iloc[-1] > data['upper_band'].iloc[-1]:
            signal = -1  # Signal de vente (prix au-dessus de la bande supérieure)
        
        return signal
    
    def macd_strategy(self, data, parameters):
        """
        Stratégie basée sur le MACD (Moving Average Convergence Divergence).
        
        Args:
            data (pd.DataFrame): Données de marché
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            int: Signal de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        # Paramètres par défaut
        fast_period = parameters.get('fast_period', 12)
        slow_period = parameters.get('slow_period', 26)
        signal_period = parameters.get('signal_period', 9)
        
        # Calcul du MACD
        data['ema_fast'] = data['close'].ewm(span=fast_period, adjust=False).mean()
        data['ema_slow'] = data['close'].ewm(span=slow_period, adjust=False).mean()
        data['macd'] = data['ema_fast'] - data['ema_slow']
        data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']
        
        # Génération du signal
        signal = 0
        
        # Vérifier s'il y a suffisamment de données
        if len(data) < slow_period + signal_period:
            self.logger.warning(f"Données insuffisantes pour la stratégie MACD")
            return signal
        
        # Vérifier le croisement du MACD et de sa ligne de signal
        if data['macd'].iloc[-2] < data['macd_signal'].iloc[-2] and data['macd'].iloc[-1] > data['macd_signal'].iloc[-1]:
            signal = 1  # Signal d'achat
        elif data['macd'].iloc[-2] > data['macd_signal'].iloc[-2] and data['macd'].iloc[-1] < data['macd_signal'].iloc[-1]:
            signal = -1  # Signal de vente
        
        return signal
    
    def ichimoku_strategy(self, data, parameters):
        """
        Stratégie basée sur l'indicateur Ichimoku Kinko Hyo.
        
        Args:
            data (pd.DataFrame): Données de marché
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            int: Signal de trading (1 pour achat, -1 pour vente, 0 pour conserver)
        """
        # Paramètres par défaut
        tenkan_period = parameters.get('tenkan_period', 9)
        kijun_period = parameters.get('kijun_period', 26)
        senkou_span_b_period = parameters.get('senkou_span_b_period', 52)
        
        # Calcul des composants Ichimoku
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period_high = data['high'].rolling(window=tenkan_period).max()
        period_low = data['low'].rolling(window=tenkan_period).min()
        data['tenkan_sen'] = (period_high + period_low) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period_high = data['high'].rolling(window=kijun_period).max()
        period_low = data['low'].rolling(window=kijun_period).min()
        data['kijun_sen'] = (period_high + period_low) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        data['senkou_span_a'] = ((data['tenkan_sen'] + data['kijun_sen']) / 2).shift(kijun_period)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period_high = data['high'].rolling(window=senkou_span_b_period).max()
        period_low = data['low'].rolling(window=senkou_span_b_period).min()
        data['senkou_span_b'] = ((period_high + period_low) / 2).shift(kijun_period)
        
        # Chikou Span (Lagging Span): Close price shifted backwards
        data['chikou_span'] = data['close'].shift(-kijun_period)
        
        # Génération du signal
        signal = 0
        
        # Vérifier s'il y a suffisamment de données
        if len(data) < senkou_span_b_period + kijun_period:
            self.logger.warning(f"Données insuffisantes pour la stratégie Ichimoku")
            return signal
        
        # Vérifier les conditions de croisement et de position par rapport au nuage
        if (data['tenkan_sen'].iloc[-2] < data['kijun_sen'].iloc[-2] and 
            data['tenkan_sen'].iloc[-1] > data['kijun_sen'].iloc[-1] and 
            data['close'].iloc[-1] > data['senkou_span_a'].iloc[-1] and 
            data['close'].iloc[-1] > data['senkou_span_b'].iloc[-1]):
            signal = 1  # Signal d'achat
        elif (data['tenkan_sen'].iloc[-2] > data['kijun_sen'].iloc[-2] and 
              data['tenkan_sen'].iloc[-1] < data['kijun_sen'].iloc[-1] and 
              data['close'].iloc[-1] < data['senkou_span_a'].iloc[-1] and 
              data['close'].iloc[-1] < data['senkou_span_b'].iloc[-1]):
            signal = -1  # Signal de vente
        
        return signal