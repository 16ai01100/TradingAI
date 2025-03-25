import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading

class WebInterfaceDesigner:
    """
    Module d'interface web pour visualiser les données de marché, les résultats de backtesting
    et contrôler les stratégies de trading.
    """
    
    def __init__(self, web_framework='flask', port=5000):
        """
        Initialise l'interface web.
        
        Args:
            web_framework (str): Framework web à utiliser ('flask' ou 'dash')
            port (int): Port sur lequel l'application web sera exécutée
        """
        self.web_framework = web_framework
        self.port = port
        self.app = None
        self.data_dir = os.path.join(os.getcwd(), 'data')
        self.results_dir = os.path.join(os.getcwd(), 'results')
        self.models_dir = os.path.join(os.getcwd(), 'models')
        self.static_dir = os.path.join(os.getcwd(), 'static')
        self.templates_dir = os.path.join(os.getcwd(), 'templates')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'web_interface_designer.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('WebInterfaceDesigner')
        self.logger.info(f"Module d'interface web initialisé avec {web_framework}")
        
        # Initialisation de l'application web
        self._initialize_app()
    
    def _initialize_app(self):
        """
        Initialise l'application web selon le framework choisi.
        """
        if self.web_framework.lower() == 'flask':
            self.app = Flask(__name__, 
                            static_folder=self.static_dir,
                            template_folder=self.templates_dir)
            self._setup_flask_routes()
        else:
            self.logger.error(f"Framework web {self.web_framework} non supporté")
            raise ValueError(f"Framework web {self.web_framework} non supporté")
    
    def _setup_flask_routes(self):
        """
        Configure les routes pour l'application Flask.
        """
        app = self.app
        
        @app.route('/')
        def index():
            return render_template('index.html')
        
        @app.route('/dashboard')
        def dashboard():
            return render_template('dashboard.html')
        
        @app.route('/strategies')
        def strategies():
            return render_template('strategies.html')
        
        @app.route('/backtesting')
        def backtesting():
            return render_template('backtesting.html')
        
        @app.route('/ml_models')
        def ml_models():
            return render_template('ml_models.html')
        
        @app.route('/settings')
        def settings():
            return render_template('settings.html')
        
        @app.route('/api/market_data')
        def api_market_data():
            symbol = request.args.get('symbol', 'BTCUSDT')
            interval = request.args.get('interval', '1d')
            limit = request.args.get('limit', 100, type=int)
            
            # Ici, vous devriez récupérer les données de marché
            # Pour l'exemple, nous retournons des données fictives
            data = self._get_market_data(symbol, interval, limit)
            return jsonify(data)
        
        @app.route('/api/strategies')
        def api_strategies():
            # Récupérer la liste des stratégies disponibles
            strategies = self._get_available_strategies()
            return jsonify(strategies)
        
        @app.route('/api/activate_strategy', methods=['POST'])
        def api_activate_strategy():
            data = request.json
            strategy_name = data.get('strategy_name')
            crypto_pair = data.get('crypto_pair')
            parameters = data.get('parameters', {})
            
            # Ici, vous devriez activer la stratégie
            success = self._activate_strategy(strategy_name, crypto_pair, parameters)
            return jsonify({'success': success})
        
        @app.route('/api/backtest_results')
        def api_backtest_results():
            # Récupérer les résultats de backtesting
            results = self._get_backtest_results()
            return jsonify(results)
        
        @app.route('/api/ml_models')
        def api_ml_models():
            # Récupérer la liste des modèles ML disponibles
            models = self._get_ml_models()
            return jsonify(models)
        
        @app.route('/api/account_info')
        def api_account_info():
            # Récupérer les informations du compte
            account_info = self._get_account_info()
            return jsonify(account_info)
    
    def _get_market_data(self, symbol, interval, limit):
        """
        Récupère les données de marché pour un symbole et un intervalle donnés.
        
        Args:
            symbol (str): Symbole de la paire de trading
            interval (str): Intervalle de temps
            limit (int): Nombre de points de données à récupérer
            
        Returns:
            dict: Données de marché
        """
        # Recherche des fichiers correspondant au symbole dans le répertoire de données
        files = [f for f in os.listdir(self.data_dir) if f.startswith(symbol) and interval in f and f.endswith('.csv')]
        
        if not files:
            self.logger.warning(f"Aucun fichier de données trouvé pour {symbol} avec intervalle {interval}")
            # Retourner des données fictives pour l'exemple
            return {
                'timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S') for _ in range(limit)],
                'open': [np.random.randint(9000, 11000) for _ in range(limit)],
                'high': [np.random.randint(9000, 11000) for _ in range(limit)],
                'low': [np.random.randint(9000, 11000) for _ in range(limit)],
                'close': [np.random.randint(9000, 11000) for _ in range(limit)],
                'volume': [np.random.randint(100, 1000) for _ in range(limit)]
            }
        
        # Utiliser le fichier le plus récent
        file_path = os.path.join(self.data_dir, sorted(files)[-1])
        
        try:
            # Chargement des données
            data = pd.read_csv(file_path)
            
            # Limiter le nombre de points de données
            if len(data) > limit:
                data = data.tail(limit)
            
            # Conversion en dictionnaire pour JSON
            result = {
                'timestamp': data['timestamp'].tolist() if 'timestamp' in data.columns else data.index.tolist(),
                'open': data['open'].tolist() if 'open' in data.columns else [],
                'high': data['high'].tolist() if 'high' in data.columns else [],
                'low': data['low'].tolist() if 'low' in data.columns else [],
                'close': data['close'].tolist() if 'close' in data.columns else [],
                'volume': data['volume'].tolist() if 'volume' in data.columns else []
            }
            
            return result
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            return {}
    
    def _get_available_strategies(self):
        """
        Récupère la liste des stratégies disponibles.
        
        Returns:
            list: Liste des stratégies disponibles
        """
        # Pour l'exemple, nous retournons une liste de stratégies fictives
        return [
            {
                'name': 'moving_average_crossover',
                'description': 'Stratégie de croisement de moyennes mobiles',
                'parameters': {
                    'short_window': {'type': 'int', 'default': 50, 'min': 5, 'max': 200},
                    'long_window': {'type': 'int', 'default': 200, 'min': 20, 'max': 500}
                }
            },
            {
                'name': 'rsi',
                'description': 'Stratégie basée sur l\'indice de force relative (RSI)',
                'parameters': {
                    'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 30},
                    'oversold': {'type': 'int', 'default': 30, 'min': 10, 'max': 40},
                    'overbought': {'type': 'int', 'default': 70, 'min': 60, 'max': 90}
                }
            },
            {
                'name': 'bollinger_bands',
                'description': 'Stratégie basée sur les bandes de Bollinger',
                'parameters': {
                    'window': {'type': 'int', 'default': 20, 'min': 5, 'max': 50},
                    'num_std': {'type': 'float', 'default': 2.0, 'min': 1.0, 'max': 3.0}
                }
            },
            {
                'name': 'macd',
                'description': 'Stratégie basée sur le MACD',
                'parameters': {
                    'fast_period': {'type': 'int', 'default': 12, 'min': 5, 'max': 30},
                    'slow_period': {'type': 'int', 'default': 26, 'min': 15, 'max': 50},
                    'signal_period': {'type': 'int', 'default': 9, 'min': 5, 'max': 20}
                }
            },
            {
                'name': 'ichimoku',
                'description': 'Stratégie basée sur l\'indicateur Ichimoku Kinko Hyo',
                'parameters': {
                    'tenkan_period': {'type': 'int', 'default': 9, 'min': 5, 'max': 20},
                    'kijun_period': {'type': 'int', 'default': 26, 'min': 15, 'max': 50},
                    'senkou_span_b_period': {'type': 'int', 'default': 52, 'min': 30, 'max': 100}
                }
            }
        ]
    
    def _activate_strategy(self, strategy_name, crypto_pair, parameters):
        """
        Active une stratégie de trading.
        
        Args:
            strategy_name (str): Nom de la stratégie
            crypto_pair (str): Paire de crypto-monnaies
            parameters (dict): Paramètres de la stratégie
            
        Returns:
            bool: True si la stratégie est activée avec succès, False sinon
        """
        # Ici, vous devriez implémenter la logique pour activer la stratégie
        # Pour l'exemple, nous retournons simplement True
        self.logger.info(f"Activation de la stratégie {strategy_name} pour {crypto_pair} avec paramètres {parameters}")
        return True
    
    def _get_backtest_results(self):
        """
        Récupère les résultats de backtesting.
        
        Returns:
            list: Liste des résultats de backtesting
        """
        results = []
        
        # Parcourir les fichiers de résultats
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if file.endswith('.json') and not file.endswith('_optimization.json'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r') as f:
                            result = json.load(f)
                            results.append(result)
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la lecture du fichier {file_path}: {e}")
        
        return results
    
    def _get_ml_models(self):
        """
        Récupère la liste des modèles ML disponibles.
        
        Returns:
            list: Liste des modèles ML disponibles
        """
        models = []
        
        # Parcourir les fichiers de modèles
        for file in os.listdir(self.models_dir):
            if file.endswith('.h5'):
                # Extraire les informations du nom de fichier
                parts = file.split('_')
                if len(parts) >= 3:
                    crypto_pair = parts[0]
                    model_type = parts[1]
                    
                    models.append({
                        'crypto_pair': crypto_pair,
                        'model_type': model_type,
                        'file_name': file,
                        'created_at': datetime.fromtimestamp(os.path.getctime(os.path.join(self.models_dir, file))).strftime('%Y-%m-%d %H:%M:%S')
                    })
        
        return models
    
    def _get_account_info(self):
        """
        Récupère les informations du compte.
        
        Returns:
            dict: Informations du compte
        """
        # Pour l'exemple, nous retournons des informations fictives
        return {
            'balances': [
                {'asset': 'BTC', 'free': '0.1', 'locked': '0.0'},
                {'asset': 'ETH', 'free': '1.5', 'locked': '0.0'},
                {'asset': 'USDT', 'free': '5000.0', 'locked': '0.0'}
            ],
            'account_type': 'SPOT',
            'can_trade': True,
            'can_withdraw': True,
            'can_deposit': True
        }
    
    def generate_candlestick_chart(self, symbol, interval, limit=100):
        """
        Génère un graphique en chandeliers pour un symbole et un intervalle donnés.
        
        Args:
            symbol (str): Symbole de la paire de trading
            interval (str): Intervalle de temps
            limit (int): Nombre de points de données à afficher
            
        Returns:
            str: Chemin du fichier HTML généré
        """
        data = self._get_market_data(symbol, interval, limit)
        
        if not data or 'timestamp' not in data or not data['timestamp']:
            self.logger.error(f"Données insuffisantes pour générer un graphique pour {symbol}")
            return None
        
        # Création du graphique avec Plotly
        fig = go.Figure(data=[go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        )])
        
        fig.update_layout(
            title=f'{symbol} - {interval}',
            xaxis_title='Date',
            yaxis_title='Prix',
            xaxis_rangeslider_visible=False
        )
        
        # Sauvegarde du graphique en HTML
        output_file = os.path.join(self.static_dir, f"{symbol}_{interval}_candlestick.html")
        fig.write_html(output_file)
        
        return output_file
    
    def generate_strategy_performance_chart(self, strategy_name, crypto_pair):
        """
        Génère un graphique de performance pour une stratégie donnée.
        
        Args:
            strategy_name (str): Nom de la stratégie
            crypto_pair (str): Paire de crypto-monnaies
            
        Returns:
            str: Chemin du fichier HTML généré
        """
        # Recherche des fichiers de résultats correspondants
        result_files = []
        for root, dirs, files in os.walk(self.results_dir):
            for file in files:
                if file.endswith('.json') and strategy_name in file and crypto_pair in file and not file.endswith('_optimization.json'):
                    result_files.append(os.path.join(root, file))
        
        if not result_files:
            self.logger.error(f"Aucun résultat trouvé pour la stratégie {strategy_name} sur {crypto_pair}")
            return None
        
        # Utiliser le fichier le plus récent
        result_file = sorted(result_files, key=os.path.getctime)[-1]
        
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            # Charger les données de backtest
            backtest_data_file = os.path.join(os.path.dirname(result_file), 'backtest_data.csv')
            if os.path.exists(backtest_data_file):
                backtest_data = pd.read_csv(backtest_data_file)
                
                # Création du graphique avec Plotly
                fig = make_subplots(rows=2, cols=1, shared_xaxis=True, 
                                  vertical_spacing=0.1, 
                                  subplot_titles=('Prix et Signaux', 'Capital'))
                
                # Graphique des prix et signaux
                fig.add_trace(
                    go.Scatter(x=backtest_data['timestamp'] if 'timestamp' in backtest_data.columns else backtest_data.index, 
                              y=backtest_data['close'], 
                              mode='lines', 
                              name='Prix'),
                    row=1, col=1
                )
                
                # Signaux d'achat
                buy_signals = backtest_data[backtest_data['signal'] == 1]
                fig.add_trace(
                    go.Scatter(x=buy_signals['timestamp'] if 'timestamp' in buy_signals.columns else buy_signals.index, 
                              y=buy_signals['close'], 
                              mode='markers', 
                              marker=dict(color='green', size=10, symbol='triangle-up'), 
                              name='Achat'),
                    row=1, col=1
                )
                
                # Signaux de vente
                sell_signals = backtest_data[backtest_data['signal'] == -1]
                fig.add_trace(
                    go.Scatter(x=sell_signals['timestamp'] if 'timestamp' in sell_signals.columns else sell_signals.index, 
                              y=sell_signals['close'], 
                              mode='markers', 
                              marker=dict(color='red', size=10, symbol='triangle-down'), 
                              name='Vente'),
                    row=1, col=1
                )
                
                # Graphique du capital
                fig.add_trace(
                    go.Scatter(x=backtest_data['timestamp'] if 'timestamp' in backtest_data.columns else backtest_data.index, 
                              y=backtest_data['capital_after_commission'], 
                              mode='lines', 
                              name='Capital'),
                    row=2, col=1
                )
                
                fig.update_layout(
                    title=f'Performance de {strategy_name} sur {crypto_pair}',
                    xaxis_title='Date',
                    xaxis2_title='Date',
                    yaxis_title='Prix',
                    yaxis2_title='Capital',
                    height=800
                )
                
                # Sauvegarde du graphique en HTML
                output_file = os.path.join(self.static_dir, f"{crypto_pair}_{strategy_name}_performance.html")
                fig.write_html(output_file)
                
                return output_file
            else:
                self.logger.error(f"Fichier de données de backtest non trouvé: {backtest_data_file}")
                return None
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du graphique de performance: {e}")
            return None
    
    def generate_ml_prediction_chart(self, crypto_pair, model_type):
        """
        Génère un graphique de prédiction pour un modèle ML donné.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            model_type (str): Type de modèle ML
            
        Returns:
            str: Chemin du fichier HTML généré
        """
        # Recherche des fichiers de performance correspondants
        performance_file = os.path.join(self.log_dir, f"{crypto_pair}_{model_type}_performance.csv")
        
        if not os.path.exists(performance_file):
            self.logger.error(f"Fichier de performance non trouvé pour {crypto_pair} avec modèle {model_type}")
            return None
        
        try:
            # Chargement des données de performance
            performance_data = pd.read_csv(performance_file)
            
            # Création du graphique avec Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=performance_data.index,
                y=performance_data['actual'],
                mode='lines',
                name='Réel'
            ))
            
            fig.add_trace(go.Scatter(
                x=performance_data.index,
                y=performance_data['predicted'],
                mode='lines',
                name='Prédiction'
            ))
            
            fig.update_layout(
                title=f'Prédictions vs Réel pour {crypto_pair} avec {model_type}',
                xaxis_title='Temps',
                yaxis_title='Prix',
                legend=dict(x=0, y=1, traceorder='normal')
            )
            
            # Sauvegarde du graphique en HTML
            output_file = os.path.join(self.static_dir, f"{crypto_pair}_{model_type}_prediction.html")
            fig.write_html(output_file)
            
            return output_file
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du graphique de prédiction: {e}")
            return None
    
    def run(self, debug=False, threaded=True):
        """
        Lance l'application web.
        
        Args:
            debug (bool): Mode debug
            threaded (bool): Exécution dans un thread séparé
        """
        if threaded:
            thread = threading.Thread(target=self._run_app, args=(debug,))
            thread.daemon = True
            thread.start()
            self.logger.info(f"Application web lancée en arrière-plan sur http://localhost:{self.port}")
            return thread
        else:
            self._run_app(debug)
    
    def _run_app(self, debug):
        """
        Exécute l'application web.
        
        Args:
            debug (bool): Mode debug
        """
        self.logger.info(f"Démarrage de l'application web sur le port {self.port}")
        self.app.run(host='0.0.0.0', port=self.port, debug=debug)
    
    def create_templates(self):
        """
        Crée les fichiers de templates HTML pour l'interface web.
        """
        # Template de base
        base_html = '''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Application de Trading{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.3/font/bootstrap-icons.css">
    <style>
        body {
            padding-top: 56px;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .sidebar {
            position: fixed;
            top: 56px;
            bottom: 0;
            left: 0;
            z-index: 100;
            padding: 48px 0 0;
            box-shadow: inset -1px 0 0 rgba(0, 0, 0, .1);
            background-color: #f8f9fa;
        }
        .sidebar-sticky {
            position: relative;
            top: 0;
            height: calc(100vh - 48px);
            padding-top: .5rem;
            overflow-x: hidden;
            overflow-y: auto;
        }
        .main-content {
            margin-left: 240px;
            padding: 20px;
            flex: 1;
        }
        @media (max-width: 767.98px) {
            .sidebar {
                width: 100%;
                position: static;
                padding: 0;
            }
            .main-content {
                margin-left: 0;
            }
        }
    </style>
    {% block extra_css %}{% endblock %}
</head>
<body>
    <nav class="navbar navbar-expand-md navbar-dark bg-dark fixed-top">
        <div class="container-fluid">
            <a class="navbar-brand" href="/">Application de Trading</a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarCollapse">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarCollapse">
                <ul class="navbar-nav ms-auto mb-2 mb-md-0">
                    <li class="nav-item">
                        <a class="nav-link" href="/">Accueil</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/dashboard">Tableau de bord</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/strategies">Stratégies</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/backtesting">Backtesting</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/ml_models">Modèles ML</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="/settings">Paramètres</a>
                    </li>
                </ul>
            </div>
        </div>
    </nav>

    <div class="container-fluid">
        <div class="row">
            <nav id="sidebarMenu" class="col-md-3 col-lg-2 d-md-block sidebar collapse">
                <div class="sidebar-sticky pt-3">
                    <ul class="nav flex-column">
                        <li class="nav-item">
                            <a class="nav-link" href="/">
                                <i class="bi bi-house"></i> Accueil
                            </a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="/dashboard">
                                <i class="bi bi-speedometer2"></i> Tableau de bord
                            </a>
                        </li>