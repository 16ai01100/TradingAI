import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.enums import *

class BinanceConnector:
    """
    Module de connexion à l'API Binance pour exécuter des opérations de trading en temps réel.
    """
    
    def __init__(self, api_key=None, api_secret=None, testnet=True):
        """
        Initialise le connecteur Binance.
        
        Args:
            api_key (str): Clé API Binance
            api_secret (str): Secret API Binance
            testnet (bool): Utiliser le réseau de test Binance (True) ou le réseau principal (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.order_history = []
        self.trade_history = []
        self.balance_history = []
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création du répertoire de logs s'il n'existe pas
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'binance_connector.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BinanceConnector')
        
        # Initialisation du client Binance si les clés API sont fournies
        if api_key and api_secret:
            self.connect()
    
    def connect(self):
        """
        Établit une connexion à l'API Binance.
        
        Returns:
            bool: True si la connexion est établie avec succès, False sinon
        """
        try:
            self.client = Client(self.api_key, self.api_secret, testnet=self.testnet)
            # Vérifier la connexion en récupérant les informations du compte
            account_info = self.client.get_account()
            self.logger.info(f"Connexion à Binance établie avec succès. Mode testnet: {self.testnet}")
            return True
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la connexion à Binance: {e}")
            return False
    
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
            account_info = self.client.get_account()
            self.logger.info("Informations du compte récupérées avec succès")
            return account_info
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des informations du compte: {e}")
            return None
    
    def get_balances(self, min_balance=0.0):
        """
        Récupère les soldes du compte.
        
        Args:
            min_balance (float): Solde minimum pour filtrer les actifs
            
        Returns:
            list: Liste des soldes
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            account_info = self.client.get_account()
            balances = account_info['balances']
            
            # Filtrer les soldes non nuls ou supérieurs au minimum spécifié
            filtered_balances = [b for b in balances if float(b['free']) > min_balance or float(b['locked']) > min_balance]
            
            # Enregistrer l'historique des soldes
            self.balance_history.append({
                'timestamp': datetime.now(),
                'balances': filtered_balances
            })
            
            self.logger.info(f"Soldes récupérés: {len(filtered_balances)} actifs")
            return filtered_balances
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des soldes: {e}")
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
            balance = self.client.get_asset_balance(asset=asset)
            self.logger.info(f"Solde de {asset} récupéré: {balance['free']} (libre), {balance['locked']} (bloqué)")
            return balance
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération du solde de {asset}: {e}")
            return None
    
    def get_symbol_info(self, symbol):
        """
        Récupère les informations sur un symbole.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            
        Returns:
            dict: Informations sur le symbole
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            symbol_info = self.client.get_symbol_info(symbol)
            self.logger.info(f"Informations sur {symbol} récupérées")
            return symbol_info
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des informations sur {symbol}: {e}")
            return None
    
    def place_market_order(self, symbol, side, quantity):
        """
        Place un ordre au marché.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            side (str): Côté de l'ordre ('BUY' ou 'SELL')
            quantity (float): Quantité à acheter/vendre
            
        Returns:
            dict: Informations sur l'ordre placé
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            # Vérifier les informations du symbole pour les règles de quantité
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Impossible de récupérer les informations pour {symbol}")
                return None
            
            # Arrondir la quantité selon les règles du symbole
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                quantity = round(quantity, precision)
            
            self.logger.info(f"Placement d'un ordre {side} au marché pour {quantity} {symbol}")
            
            # Placer l'ordre
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            
            # Enregistrer l'ordre dans l'historique
            self.order_history.append({
                'timestamp': datetime.now(),
                'order': order
            })
            
            self.logger.info(f"Ordre placé avec succès: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors du placement de l'ordre: {e}")
            return None
    
    def place_limit_order(self, symbol, side, quantity, price):
        """
        Place un ordre limite.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            side (str): Côté de l'ordre ('BUY' ou 'SELL')
            quantity (float): Quantité à acheter/vendre
            price (float): Prix limite
            
        Returns:
            dict: Informations sur l'ordre placé
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            # Vérifier les informations du symbole pour les règles de quantité et de prix
            symbol_info = self.client.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.error(f"Impossible de récupérer les informations pour {symbol}")
                return None
            
            # Arrondir la quantité selon les règles du symbole
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                quantity = round(quantity, precision)
            
            # Arrondir le prix selon les règles du symbole
            price_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER'), None)
            if price_filter:
                tick_size = float(price_filter['tickSize'])
                price_precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                price = round(price, price_precision)
            
            self.logger.info(f"Placement d'un ordre {side} limite à {price} pour {quantity} {symbol}")
            
            # Placer l'ordre
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=quantity,
                price=str(price)
            )
            
            # Enregistrer l'ordre dans l'historique
            self.order_history.append({
                'timestamp': datetime.now(),
                'order': order
            })
            
            self.logger.info(f"Ordre limite placé avec succès: {order['orderId']}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors du placement de l'ordre limite: {e}")
            return None
    
    def cancel_order(self, symbol, order_id):
        """
        Annule un ordre.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            order_id (int): ID de l'ordre à annuler
            
        Returns:
            dict: Informations sur l'annulation
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Annulation de l'ordre {order_id} pour {symbol}")
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Ordre annulé avec succès: {result}")
            return result
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de l'annulation de l'ordre: {e}")
            return None
    
    def get_open_orders(self, symbol=None):
        """
        Récupère les ordres ouverts.
        
        Args:
            symbol (str, optional): Symbole de la paire de trading (ex: 'BTCUSDT')
                                   Si None, récupère tous les ordres ouverts
            
        Returns:
            list: Liste des ordres ouverts
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            if symbol:
                self.logger.info(f"Récupération des ordres ouverts pour {symbol}")
                open_orders = self.client.get_open_orders(symbol=symbol)
            else:
                self.logger.info("Récupération de tous les ordres ouverts")
                open_orders = self.client.get_open_orders()
            
            self.logger.info(f"Ordres ouverts récupérés: {len(open_orders)}")
            return open_orders
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des ordres ouverts: {e}")
            return None
    
    def get_order_status(self, symbol, order_id):
        """
        Récupère le statut d'un ordre.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            order_id (int): ID de l'ordre
            
        Returns:
            dict: Informations sur l'ordre
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération du statut de l'ordre {order_id} pour {symbol}")
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            self.logger.info(f"Statut de l'ordre récupéré: {order['status']}")
            return order
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération du statut de l'ordre: {e}")
            return None
    
    def get_all_orders(self, symbol, limit=500):
        """
        Récupère tous les ordres pour un symbole donné.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            limit (int): Nombre maximum d'ordres à récupérer
            
        Returns:
            list: Liste des ordres
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération de tous les ordres pour {symbol}")
            orders = self.client.get_all_orders(symbol=symbol, limit=limit)
            self.logger.info(f"Ordres récupérés: {len(orders)}")
            return orders
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des ordres: {e}")
            return None
    
    def get_my_trades(self, symbol, limit=500):
        """
        Récupère les transactions de l'utilisateur pour un symbole donné.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            limit (int): Nombre maximum de transactions à récupérer
            
        Returns:
            list: Liste des transactions
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération des transactions pour {symbol}")
            trades = self.client.get_my_trades(symbol=symbol, limit=limit)
            
            # Enregistrer les transactions dans l'historique
            for trade in trades:
                self.trade_history.append({
                    'timestamp': datetime.fromtimestamp(trade['time'] / 1000),
                    'trade': trade
                })
            
            self.logger.info(f"Transactions récupérées: {len(trades)}")
            return trades
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des transactions: {e}")
            return None
    
    def calculate_position_size(self, symbol, risk_percentage=1.0, stop_loss_percentage=2.0):
        """
        Calcule la taille de position en fonction du risque et du stop loss.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            risk_percentage (float): Pourcentage du capital à risquer
            stop_loss_percentage (float): Pourcentage de stop loss
            
        Returns:
            float: Taille de la position
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return 0.0
        
        try:
            # Récupérer le solde en USDT
            usdt_balance = float(self.get_asset_balance('USDT')['free'])
            
            # Calculer le montant à risquer
            risk_amount = usdt_balance * (risk_percentage / 100.0)
            
            # Récupérer le prix actuel
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # Calculer la taille de position en fonction du stop loss
            position_size = risk_amount / (current_price * (stop_loss_percentage / 100.0))
            
            # Arrondir à la précision requise
            symbol_info = self.client.get_symbol_info(symbol)
            lot_size_filter = next((f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            
            if lot_size_filter:
                step_size = float(lot_size_filter['stepSize'])
                precision = len(str(step_size).split('.')[-1].rstrip('0'))
                position_size = round(position_size, precision)
            
            self.logger.info(f"Taille de position calculée pour {symbol}: {position_size} (risque: {risk_percentage}%, stop loss: {stop_loss_percentage}%)")
            return position_size
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul de la taille de position: {e}")
            return 0.0
    
    def get_performance_metrics(self, symbol=None, start_time=None, end_time=None):
        """
        Calcule les métriques de performance du trading.
        
        Args:
            symbol (str, optional): Symbole de la paire de trading (ex: 'BTCUSDT')
            start_time (datetime, optional): Date de début
            end_time (datetime, optional): Date de fin
            
        Returns:
            dict: Métriques de performance
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            # Définir les dates par défaut si non spécifiées
            if start_time is None:
                start_time = datetime.now() - timedelta(days=30)
            if end_time is None:
                end_time = datetime.now()
            
            # Convertir les dates en timestamps
            start_timestamp = int(start_time.timestamp() * 1000)
            end_timestamp = int(end_time.timestamp() * 1000)
            
            # Récupérer les transactions
            if symbol:
                trades = self.client.get_my_trades(symbol=symbol)
            else:
                # Pour tous les symboles, il faut les récupérer un par un
                # On se limite aux paires USDT les plus courantes
                common_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT']
                trades = []
                for sym in common_symbols:
                    try:
                        sym_trades = self.client.get_my_trades(symbol=sym)
                        trades.extend(sym_trades)
                    except:
                        pass
            
            # Filtrer par date
            filtered_trades = [t for t in trades if t['time'] >= start_timestamp and t['time'] <= end_timestamp]
            
            if not filtered_trades:
                self.logger.warning("Aucune transaction trouvée pour la période spécifiée")
                return {
                    'total_trades': 0,
                    'win_rate': 0.0,
                    'profit_loss': 0.0,
                    'avg_profit_per_trade': 0.0,
                    'max_drawdown': 0.0
                }
            
            # Calculer les métriques
            buy_trades = [t for t in filtered_trades if t['isBuyer']]
            sell_trades = [t for t in filtered_trades if not t['isBuyer']]
            
            # Calculer le profit/perte (simplifié)
            total_buy = sum(float(t['price']) * float(t['qty']) for t in buy_trades)
            total_sell = sum(float(t['price']) * float(t['qty']) for t in sell_trades)
            profit_loss = total_sell - total_buy
            
            # Calculer le taux de réussite (simplifié)
            # On considère qu'une transaction est gagnante si le prix de vente > prix d'achat
            # Cette logique est simplifiée et devrait être améliorée dans une implémentation réelle
            winning_trades = 0
            losing_trades = 0
            
            # Simplification: on suppose que les achats et ventes sont appariés chronologiquement
            buy_prices = {}
            for trade in filtered_trades:
                symbol = trade['symbol']
                if trade['isBuyer']:
                    if symbol not in buy_prices:
                        buy_prices[symbol] = []
                    buy_prices[symbol].append(float(trade['price']))
                else:  # vente
                    if symbol in buy_prices and buy_prices[symbol]:
                        buy_price = buy_prices[symbol].pop(0)  # FIFO
                        if float(trade['price']) > buy_price:
                            winning_trades += 1
                        else:
                            losing_trades += 1
            
            total_trades = winning_trades + losing_trades
            win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
            
            # Calculer le profit moyen par transaction
            avg_profit = profit_loss / total_trades if total_trades > 0 else 0.0
            
            # Calculer le drawdown maximum (simplifié)
            # Dans une implémentation réelle, cela nécessiterait un suivi plus détaillé de l'équité
            max_drawdown = 0.0  # Placeholder
            
            metrics = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_loss': profit_loss,
                'avg_profit_per_trade': avg_profit,
                'max_drawdown': max_drawdown
            }
            
            self.logger.info(f"Métriques de performance calculées: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Erreur lors du calcul des métriques de performance: {e}")
            return None
    
    def get_market_data(self, symbol, interval='1h', limit=500):
        """
        Récupère les données de marché pour un symbole donné.
        
        Args:
            symbol (str): Symbole de la paire de trading (ex: 'BTCUSDT')
            interval (str): Intervalle de temps (ex: '1h', '1d')
            limit (int): Nombre maximum de bougies à récupérer
            
        Returns:
            pd.DataFrame: DataFrame contenant les données de marché
        """
        if not self.client:
            self.logger.error("Client Binance non initialisé. Veuillez vous connecter d'abord.")
            return None
        
        try:
            self.logger.info(f"Récupération des données de marché pour {symbol} avec intervalle {interval}")
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            
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
            return data
        except BinanceAPIException as e:
            self.logger.error(f"Erreur lors de la récupération des données de marché: {e}")
            return None