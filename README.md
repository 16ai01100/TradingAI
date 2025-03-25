# Modules de Trading Avancés

Ce répertoire contient les modules de l'application de trading modulaire avec apprentissage profond. Les modules récemment ajoutés incluent une gestion des risques avancée et une analyse multi-fréquence.

## Modules Principaux

### Gestion des Risques (`risk_manager.py`)

Le module `RiskManager` implémente des fonctionnalités avancées de gestion des risques pour le système de trading :

- **Value-at-Risk (VaR)** : Calcul de la perte potentielle maximale avec un niveau de confiance donné
- **Conditional Value-at-Risk (CVaR)** : Mesure de la perte moyenne au-delà de la VaR
- **Surveillance de liquidité** : Évaluation de la liquidité du marché et de l'impact potentiel des transactions
- **Ajustement dynamique des positions** : Adaptation de la taille des positions en fonction des métriques de risque
- **Optimisation de portefeuille** : Calcul des allocations optimales selon la théorie moderne du portefeuille
- **Coûts de transaction** : Estimation et intégration des coûts de transaction dans les décisions
- **Génération de rapports** : Création de rapports détaillés sur les métriques de risque

#### Exemple d'utilisation

```python
# Initialiser le gestionnaire de risques
risk_manager = RiskManager(
    confidence_level=0.95,
    var_window=252,
    max_drawdown_limit=0.15,
    max_position_size=0.05,
    max_leverage=1.5
)

# Calculer la VaR et CVaR
var_95 = risk_manager.calculate_var(returns, 0.95)
cvar_95 = risk_manager.calculate_cvar(returns, 0.95)

# Ajuster la taille de position
adjusted_size = risk_manager.adjust_position_size(
    crypto_pair, base_position_size, market_data, risk_metrics
)
```

### Analyse Multi-Fréquence (`multi_frequency_analyzer.py`)

Le module `MultiFrequencyAnalyzer` permet d'analyser les données de marché à travers différentes échelles temporelles et sources :

- **Analyse multi-timeframes** : Combinaison de données de différentes temporalités (1m, 5m, 15m, 1h, 4h, 1d)
- **Intégration de données on-chain** : Incorporation de métriques blockchain dans l'analyse
- **Détection de patterns cross-fréquence** : Identification de divergences, confirmations de tendance, etc.
- **Génération de signaux combinés** : Fusion des signaux de différentes fréquences et stratégies
- **Pondération adaptative** : Ajustement des poids des différentes fréquences en fonction des performances

#### Exemple d'utilisation

```python
# Initialiser l'analyseur multi-fréquence
multi_freq_analyzer = MultiFrequencyAnalyzer(
    data_collector=data_collector,
    binance_connector=binance_connector,
    timeframes=['1m', '5m', '15m', '1h', '4h', '1d']
)

# Récupérer des données multi-timeframes
multi_timeframe_data = multi_freq_analyzer.fetch_multi_timeframe_data(crypto_pair)

# Analyser les patterns cross-fréquence
patterns = multi_freq_analyzer.analyze_cross_frequency_patterns(crypto_pair, multi_timeframe_data)

# Générer un signal combiné
combined_signal = multi_freq_analyzer.generate_combined_signal(crypto_pair, strategy_signals)
```

## Intégration avec les Modules Existants

Les nouveaux modules s'intègrent parfaitement avec les modules existants :

- `RiskManager` peut être utilisé avec `StrategyExecutor` pour ajuster les tailles de position
- `MultiFrequencyAnalyzer` peut être utilisé avec `StrategySelector` pour améliorer la sélection de stratégies
- Les deux modules peuvent être utilisés avec `BacktestingEngineer` pour des backtests plus réalistes

## Exemples

Des exemples d'utilisation sont disponibles dans le répertoire `examples/` :

- `risk_management_example.py` : Démonstration des fonctionnalités de gestion des risques
- `multi_frequency_analysis_example.py` : Démonstration de l'analyse multi-fréquence

## Bonnes Pratiques

1. **Séparation des composants** : Maintenir une séparation claire entre les composants (Data, Signal, Risk, Execution)
2. **Injection de dépendances** : Utiliser l'injection de dépendances via les constructeurs
3. **Logging complet** : Activer le logging pour toutes les opérations critiques
4. **Gestion des erreurs** : Implémenter une gestion robuste des erreurs et des reprises sur incident
5. **Tests unitaires** : Créer des tests pour chaque fonctionnalité
