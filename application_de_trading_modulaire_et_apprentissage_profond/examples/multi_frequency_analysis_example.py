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
    DataCollector, BinanceConnector, StrategyExecutor, StrategySelector,
    MultiFrequencyAnalyzer, RiskManager
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
    
    # Initialiser le sélecteur de stratégies
    strategy_selector = StrategySelector()
    
    # Initialiser l'analyseur multi-fréquence
    multi_freq_analyzer = MultiFrequencyAnalyzer(
        data_collector=data_collector,
        binance_connector=binance_connector,
        timeframes=['1m', '5m', '15m', '1h', '4h', '1d'],
        correlation_threshold=0.7
    )
    
    # Initialiser le gestionnaire de risques
    risk_manager = RiskManager()
    
    # Paire de crypto-monnaie à analyser
    crypto_pair = 'BTCUSDT'
    
    print(f"\nAnalyse multi-fréquence pour {crypto_pair}...")
    
    # Simuler des données multi-timeframes
    multi_timeframe_data = simulate_multi_timeframe_data()
    
    # Analyser les patterns cross-fréquence
    print("Analyse des patterns cross-fréquence...")
    patterns = multi_freq_analyzer.analyze_cross_frequency_patterns(crypto_pair, multi_timeframe_data)
    
    # Afficher les divergences détectées
    if 'divergences' in patterns:
        print("\nDivergences détectées:")
        for div_key, div_info in patterns['divergences'].items():
            print(f"  {div_key}: Type {div_info['type']}, Force {div_info['strength']:.4f}")
    
    # Afficher les confirmations de tendance
    if 'trend_confirmations' in patterns and 'overall' in patterns['trend_confirmations']:
        conf_info = patterns['trend_confirmations']['overall']
        print("\nConfirmation de tendance:")
        print(f"  Type: {conf_info['type']}")
        print(f"  Force: {conf_info['strength']:.4f}")
        print(f"  Compteurs: Haussier {conf_info['bullish_count']}, Baissier {conf_info['bearish_count']}, Neutre {conf_info['neutral_count']}")
    
    # Afficher les niveaux de support/résistance
    if 'support_resistance' in patterns:
        sr_levels = patterns['support_resistance']
        print("\nNiveaux de support/résistance:")
        if 'supports' in sr_levels and sr_levels['supports']:
            print(f"  Supports: {', '.join([f'{s:.2f}' for s in sr_levels['supports'][:3]])}")
        if 'resistances' in sr_levels and sr_levels['resistances']:
            print(f"  Résistances: {', '.join([f'{r:.2f}' for r in sr_levels['resistances'][:3]])}")
    
    # Générer un signal combiné
    print("\nGénération du signal combiné...")
    
    # Simuler des signaux de stratégies
    strategy_signals = {
        'moving_average_crossover': 1,  # Signal d'achat
        'rsi': 0,                      # Signal neutre
        'bollinger_bands': 1,          # Signal d'achat
        'macd': -1,                    # Signal de vente
        'ichimoku': 1                  # Signal d'achat
    }
    
    # Générer le signal combiné
    combined_signal = multi_freq_analyzer.generate_combined_signal(crypto_pair, strategy_signals)
    
    print(f"Signal combiné: {combined_signal['signal']} (force: {combined_signal['strength']:.4f}, confiance: {combined_signal['confidence']:.4f})")
    print(f"Score haussier: {combined_signal['bullish_score']:.4f}, Score baissier: {combined_signal['bearish_score']:.4f}")
    
    # Intégration avec le gestionnaire de risques
    print("\nIntégration avec le gestionnaire de risques...")
    
    # Simuler des données de marché et métriques de risque
    market_data = {
        'price': 40000,
        'spread': 0.001,
        'liquidity_impact': 0.002
    }
    
    risk_metrics = {
        'var': 0.03,
        'volatility': 0.02,
        'liquidity_impact': 0.01,
        'current_exposure': 0.5
    }
    
    # Calculer la taille de position de base
    base_position_size = 0.1  # 10% du capital
    
    # Ajuster la taille de position en fonction des métriques de risque
    adjusted_size = risk_manager.adjust_position_size(crypto_pair, base_position_size, market_data, risk_metrics)
    
    print(f"Taille de position de base: {base_position_size:.4f}")
    print(f"Taille de position ajustée: {adjusted_size:.4f}")
    
    # Simuler des données de carnet d'ordres
    order_book_data = {
        'bids': [(39900, 2.5), (39800, 5.0), (39700, 7.5)],
        'asks': [(40100, 2.0), (40200, 4.0), (40300, 6.0)]
    }
    
    # Surveiller la liquidité
    liquidity_metrics = risk_manager.monitor_liquidity(crypto_pair, order_book_data, adjusted_size)
    
    if liquidity_metrics:
        print("\nMétriques de liquidité:")
        print(f"  Spread: {liquidity_metrics['spread']:.6f}")
        print(f"  Profondeur totale: {liquidity_metrics['total_depth']:.2f}")
        print(f"  Ratio d'impact: {liquidity_metrics['impact_ratio']:.6f}")
        print(f"  Slippage estimé: {liquidity_metrics['estimated_slippage']:.6f}")
    
    # Calculer les coûts de transaction
    transaction_costs = risk_manager.calculate_transaction_costs(crypto_pair, adjusted_size, 'market', market_data)
    
    print("\nCoûts de transaction:")
    print(f"  Frais de base: {transaction_costs['base_fee']:.4f}")
    print(f"  Slippage: {transaction_costs['slippage']:.4f}")
    print(f"  Coût total: {transaction_costs['total_cost']:.4f} ({transaction_costs['cost_percentage']:.4f}%)")
    
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