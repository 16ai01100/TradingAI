import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Ajouter le chemin du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Importer les modules nécessaires
from application_de_trading_modulaire_et_apprentissage_profond.modules import (
    DataCollector, BinanceConnector, StrategyExecutor, RiskManager, MultiFrequencyAnalyzer
)

def main():
    # Initialisation des modules
    print("Initialisation des modules...")
    
    # Clés API fictives pour l'exemple
    api_key = "votre_api_key"
    api_secret = "votre_api_secret"
    
    # Initialiser le collecteur de données
    data_collector = DataCollector(api_key=api_key, api_secret=api_secret)
    
    # Initialiser le connecteur Binance
    binance_connector = BinanceConnector(api_key=api_key, api_secret=api_secret, testnet=True)
    
    # Initialiser l'exécuteur de stratégies
    strategy_executor = StrategyExecutor(api_key=api_key, api_secret=api_secret)
    
    # Initialiser le gestionnaire de risques avec des paramètres personnalisés
    risk_manager = RiskManager(
        confidence_level=0.95,
        var_window=252,
        max_drawdown_limit=0.15,
        max_position_size=0.05,
        max_leverage=1.5,
        rebalance_threshold=0.03
    )
    
    # Initialiser l'analyseur multi-fréquence
    multi_freq_analyzer = MultiFrequencyAnalyzer(
        data_collector=data_collector,
        binance_connector=binance_connector,
        timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
        correlation_threshold=0.7
    )
    
    # Paires de crypto-monnaies à analyser
    crypto_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
    
    # Exemple d'utilisation du gestionnaire de risques
    print("\nDémonstration du gestionnaire de risques...")
    
    # Simuler des rendements historiques
    np.random.seed(42)  # Pour la reproductibilité
    returns = pd.Series(np.random.normal(0.0005, 0.02, 252))  # 252 jours de trading
    
    # Calculer la VaR et CVaR
    var_95 = risk_manager.calculate_var(returns, 0.95)
    cvar_95 = risk_manager.calculate_cvar(returns, 0.95)
    
    print(f"Value-at-Risk (95%): {var_95:.4f}")
    print(f"Conditional Value-at-Risk (95%): {cvar_95:.4f}")
    
    # Simuler un portefeuille avec plusieurs actifs
    num_assets = 3
    weights = np.array([0.5, 0.3, 0.2])  # Poids des actifs
    expected_returns = np.array([0.001, 0.0015, 0.002])  # Rendements attendus
    
    # Matrice de covariance
    cov_matrix = np.array([
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0005, 0.0002],
        [0.0001, 0.0002, 0.0006]
    ])
    
    # Calculer la VaR du portefeuille
    portfolio_var = risk_manager.calculate_portfolio_var(weights, cov_matrix, expected_returns)
    print(f"VaR du portefeuille: {portfolio_var:.4f}")
    
    # Calculer le portefeuille optimal
    optimal_portfolio = risk_manager.calculate_optimal_portfolio(expected_returns, cov_matrix)
    print("\nPortefeuille optimal:")
    print(f"Poids: {optimal_portfolio['weights']}")
    print(f"Rendement attendu: {optimal_portfolio['return']:.4f}")
    print(f"Volatilité: {optimal_portfolio['volatility']:.4f}")
    print(f"Ratio de Sharpe: {optimal_portfolio['sharpe_ratio']:.4f}")
    
    # Exemple d'utilisation de l'analyseur multi-fréquence
    print("\nDémonstration de l'analyseur multi-fréquence...")
    
    # Récupérer des données multi-timeframes (simulation)
    print("Récupération des données multi-timeframes...")
    
    # Simuler des données pour différents timeframes
    data = {}
    for crypto_pair in crypto_pairs:
        print(f"Analyse de {crypto_pair}...")
        
        # Dans un cas réel, on utiliserait:
        # multi_freq_data = multi_freq_analyzer.fetch_multi_timeframe_data(crypto_pair)
        # Mais pour l'exemple, on simule des données:
        multi_freq_data = simulate_multi_timeframe_data()
        
        # Analyser les patterns cross-fréquence
        patterns = multi_freq_analyzer.analyze_cross_frequency_patterns(crypto_pair, multi_freq_data)
        
        # Générer un signal combiné
        signal = multi_freq_analyzer.generate_combined_signal(crypto_pair)
        
        print(f"Signal pour {crypto_pair}: {signal['signal']} (force: {signal['strength']:.4f}, confiance: {signal['confidence']:.4f})")
    
    print("\nExemple terminé.")

def simulate_multi_timeframe_data():
    """Simule des données multi-timeframes pour l'exemple."""
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    data = {}
    
    for tf in timeframes:
        # Nombre de points de données selon le timeframe
        if tf == '1m':
            n_points = 1440  # 1 jour
        elif tf == '5m':
            n_points = 1440  # 5 jours
        elif tf == '15m':
            n_points = 960   # 10 jours
        elif tf == '1h':
            n_points = 720   # 30 jours
        elif tf == '4h':
            n_points = 360   # 60 jours
        else:  # '1d'
            n_points = 365   # 1 an
        
        # Créer un DataFrame avec des données simulées
        dates = [datetime.now() - timedelta(minutes=i) for i in range(n_points)]
        dates.reverse()
        
        # Simuler un mouvement de prix avec une tendance et du bruit
        np.random.seed(42 + len(tf))  # Différente seed pour chaque timeframe
        close = 40000 + np.cumsum(np.random.normal(0, 100, n_points))
        
        # Ajouter une tendance selon le timeframe
        if tf in ['1h', '4h', '1d']:
            close = close + np.linspace(0, 2000, n_points)  # Tendance haussière
        
        # Créer les autres colonnes
        open_price = close - np.random.normal(0, 50, n_points)
        high = np.maximum(close, open_price) + np.random.normal(50, 30, n_points)
        low = np.minimum(close, open_price) - np.random.normal(50, 30, n_points)
        volume = np.random.normal(1000, 200, n_points) * (1 + np.sin(np.linspace(0, 10, n_points)))
        
        # Créer le DataFrame
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
        
        df.set_index('timestamp', inplace=True)
        data[tf] = df
    
    return data

if __name__ == "__main__":
    main()