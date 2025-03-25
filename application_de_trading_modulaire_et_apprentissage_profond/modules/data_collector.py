import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException

class DataCollector:
    """
    Module de collecte de données pour récupérer les données de marché de Binance.
    """
    
    def __init__(self, api_key=None, api_secret=None, crypto_pairs=None):
        """
        Initialise le collecteur de données.
        
        Args:
            api_key (str): Clé API Binance
            api_secret (str): Secret API Binance
            crypto_pairs (list): Liste des paires de crypto-monnaies à surveiller
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.crypto_pairs = crypto_pairs or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.client = None
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
                logging.FileHandler(os.path.join(self.log_dir, 'data_collector.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('DataCollector')
        
        # Initialisation du client Binance si les clés API sont fournies
        if api_key and api_secret:
            self.connect_to_binance()
    
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
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Récupère les données historiques de Binance.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            interval (str): Intervalle de temps (ex: '1h', '1d')
            start_str (str): Date de début (ex: '1 Jan, 2020')
            end_str (str): Date de fin (ex: '1 Jan, 2021')
            
        Returns:
            pd.DataFrame: DataFrame contenant les données historiques
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération des données historiques pour {symbol} avec intervalle {interval}")
            klines = self.client.get_historical_klines(symbol, interval, start_str, end_str)
            
            # Conversion en DataFrame
            data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                               'close_time', 'quote_asset_volume', 'number_of_trades', 
                                               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            
            # Conversion des types de données
            data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
            data.set_index('timestamp', inplace=True)
            
            # Conversion des colonnes numériques
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, axis=1)
            
            self.logger.info(f"Données récupérées: {len(data)} entrées")
            
            # Sauvegarde des données
            file_path = os.path.join(self.data_dir, f"{symbol}_{interval}_{start_str.replace(' ', '_')}_{end_str.replace(' ', '_') if end_str else 'now'}.csv")
            data.to_csv(file_path)
            self.logger.info(f"Données sauvegardées dans {file_path}")
            
            return data
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des données: {e}")
            return None
    
    def get_recent_trades(self, symbol, limit=500):
        """
        Récupère les transactions récentes pour un symbole donné.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            limit (int): Nombre maximum de transactions à récupérer
            
        Returns:
            pd.DataFrame: DataFrame contenant les transactions récentes
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération des transactions récentes pour {symbol}")
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            
            # Conversion en DataFrame
            data = pd.DataFrame(trades)
            
            # Conversion des types de données
            data['time'] = pd.to_datetime(data['time'], unit='ms')
            data.set_index('time', inplace=True)
            
            # Conversion des colonnes numériques
            numeric_columns = ['price', 'qty', 'quoteQty']
            data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, axis=1)
            
            self.logger.info(f"Transactions récupérées: {len(data)} entrées")
            
            return data
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des transactions: {e}")
            return None
    
    def get_ticker_price(self, symbol=None):
        """
        Récupère le prix actuel d'un ou plusieurs symboles.
        
        Args:
            symbol (str, optional): Symbole de la paire de trading (ex: 'BTCUSDT')
                                   Si None, récupère les prix pour toutes les paires
            
        Returns:
            dict or pd.DataFrame: Prix actuel pour le(s) symbole(s)
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            if symbol:
                self.logger.info(f"Récupération du prix actuel pour {symbol}")
                price = self.client.get_symbol_ticker(symbol=symbol)
                return price
            else:
                self.logger.info("Récupération des prix actuels pour toutes les paires")
                prices = self.client.get_all_tickers()
                return pd.DataFrame(prices)
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des prix: {e}")
            return None
    
    def get_order_book(self, symbol, limit=100):
        """
        Récupère le carnet d'ordres pour un symbole donné.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            limit (int): Nombre maximum d'ordres à récupérer
            
        Returns:
            dict: Carnet d'ordres
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération du carnet d'ordres pour {symbol}")
            order_book = self.client.get_order_book(symbol=symbol, limit=limit)
            return order_book
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération du carnet d'ordres: {e}")
            return None
    
    def get_account_info(self):
        """
        Récupère les informations du compte.
        
        Returns:
            dict: Informations du compte
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info("Récupération des informations du compte")
            account_info = self.client.get_account()
            return account_info
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des informations du compte: {e}")
            return None
    
    def get_asset_balance(self, asset):
        """
        Récupère le solde d'un actif spécifique.
        
        Args:
            asset (str): Symbole de l'actif (ex: 'BTC')
            
        Returns:
            dict: Solde de l'actif
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération du solde pour {asset}")
            balance = self.client.get_asset_balance(asset=asset)
            return balance
        
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération du solde: {e}")
            return None
    
    def setup_data_streams(self, crypto_pairs=None):
        """
        Configure les flux de données pour les paires de crypto-monnaies spécifiées.
        
        Args:
            crypto_pairs (list, optional): Liste des paires de crypto-monnaies
            
        Returns:
            bool: True si la configuration est réussie, False sinon
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return False
        
        pairs = crypto_pairs or self.crypto_pairs
        self.logger.info(f"Configuration des flux de données pour {pairs}")
        
        # Ici, vous pouvez implémenter la logique pour configurer les websockets
        # ou d'autres mécanismes de streaming de données
        
        return True
    
    def download_all_historical_data(self, interval='1d', start_date='1 Jan, 2020', end_date=None):
        """
        Télécharge les données historiques pour toutes les paires configurées.
        
        Args:
            interval (str): Intervalle de temps (ex: '1h', '1d')
            start_date (str): Date de début (ex: '1 Jan, 2020')
            end_date (str): Date de fin (ex: '1 Jan, 2021')
            
        Returns:
            dict: Dictionnaire des DataFrames par paire
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        data_dict = {}
        for pair in self.crypto_pairs:
            self.logger.info(f"Téléchargement des données historiques pour {pair}")
            data = self.get_historical_klines(pair, interval, start_date, end_date)
            if data is not None:
                data_dict[pair] = data
            
            # Pause pour éviter de dépasser les limites de l'API
            time.sleep(1)
        
        return data_dict
    
    def visualize_price_data(self, data, title="Évolution des prix"):
        """
        Visualise les données de prix.
        
        Args:
            data (pd.DataFrame): DataFrame contenant les données de prix
            title (str): Titre du graphique
        """
        plt.figure(figsize=(12, 6))
        plt.plot(data['close'])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.grid(True)
        
        # Sauvegarde du graphique
        file_path = os.path.join(self.log_dir, f"{title.replace(' ', '_')}.png")
        plt.savefig(file_path)
        self.logger.info(f"Graphique sauvegardé dans {file_path}")
        
        plt.close()
    
    def calculate_technical_indicators(self, data):
        """
        Calcule les indicateurs techniques sur les données de prix.
        
        Args:
            data (pd.DataFrame): DataFrame contenant les données de prix
            
        Returns:
            pd.DataFrame: DataFrame avec les indicateurs techniques ajoutés
        """
        df = data.copy()
        
        # Moyennes mobiles
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA25'] = df['close'].rolling(window=25).mean()
        df['MA99'] = df['close'].rolling(window=99).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['20MA'] = df['close'].rolling(window=20).mean()
        df['Upper_Band'] = df['20MA'] + (df['close'].rolling(window=20).std() * 2)
        df['Lower_Band'] = df['20MA'] - (df['close'].rolling(window=20).std() * 2)
        
        return df