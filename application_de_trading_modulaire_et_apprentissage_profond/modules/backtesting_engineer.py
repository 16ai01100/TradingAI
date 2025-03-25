import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import json

class BacktestingEngineer:
    """
    Module de backtesting pour évaluer les performances des stratégies de trading.
    """
    
    def __init__(self, historical_data_source=None, crypto_pairs=None):
        """
        Initialise le module de backtesting.
        
        Args:
            historical_data_source (str): Source des données historiques
            crypto_pairs (list): Liste des paires de crypto-monnaies à tester
        """
        self.historical_data_source = historical_data_source or 'data'
        self.crypto_pairs = crypto_pairs or ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.data = {}
        self.results = {}
        self.data_dir = os.path.join(os.getcwd(), 'data')
        self.results_dir = os.path.join(os.getcwd(), 'results')
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'backtesting_engineer.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('BacktestingEngineer')
        self.logger.info("Module de backtesting initialisé")
    
    def load_historical_data(self, crypto_pair, file_path=None, start_date=None, end_date=None):
        """
        Charge les données historiques pour une paire de crypto-monnaies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            file_path (str): Chemin du fichier de données (si None, utilise le chemin par défaut)
            start_date (str): Date de début pour filtrer les données
            end_date (str): Date de fin pour filtrer les données
            
        Returns:
            pd.DataFrame: DataFrame contenant les données historiques
        """
        if file_path is None:
            # Recherche des fichiers correspondant à la paire dans le répertoire de données
            files = [f for f in os.listdir(self.data_dir) if f.startswith(crypto_pair) and f.endswith('.csv')]
            if not files:
                self.logger.error(f"Aucun fichier de données trouvé pour {crypto_pair}")
                return None
            
            # Utiliser le fichier le plus récent
            file_path = os.path.join(self.data_dir, sorted(files)[-1])
        
        self.logger.info(f"Chargement des données historiques depuis {file_path}")
        
        try:
            # Chargement des données
            data = pd.read_csv(file_path)
            
            # Conversion de la colonne de date en datetime si nécessaire
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data.set_index('timestamp', inplace=True)
            
            # Filtrage par date si spécifié
            if start_date:
                start_date = pd.to_datetime(start_date)
                data = data[data.index >= start_date]
            
            if end_date:
                end_date = pd.to_datetime(end_date)
                data = data[data.index <= end_date]
            
            # Stockage des données
            self.data[crypto_pair] = data
            
            self.logger.info(f"Données chargées: {len(data)} entrées")
            return data
        
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données: {e}")
            return None
    
    def run_backtest(self, strategy_func, crypto_pair, parameters=None, initial_capital=10000.0, commission=0.001):
        """
        Exécute un backtest pour une stratégie donnée sur une paire de crypto-monnaies.
        
        Args:
            strategy_func (callable): Fonction de stratégie à tester
            crypto_pair (str): Paire de crypto-monnaies
            parameters (dict): Paramètres de la stratégie
            initial_capital (float): Capital initial
            commission (float): Taux de commission
            
        Returns:
            dict: Résultats du backtest
        """
        if crypto_pair not in self.data:
            self.logger.error(f"Données non disponibles pour {crypto_pair}. Veuillez charger les données d'abord.")
            return None
        
        data = self.data[crypto_pair].copy()
        parameters = parameters or {}
        
        self.logger.info(f"Exécution du backtest pour {crypto_pair} avec paramètres {parameters}")
        
        # Génération des signaux
        signals = strategy_func(data, parameters)
        data['signal'] = signals
        
        # Calcul des positions (1 pour long, -1 pour short, 0 pour neutre)
        data['position'] = data['signal'].fillna(0)
        
        # Calcul des rendements
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['position'].shift(1) * data['returns']
        
        # Calcul des rendements cumulés
        data['cumulative_returns'] = (1 + data['returns']).cumprod()
        data['cumulative_strategy_returns'] = (1 + data['strategy_returns']).cumprod()
        
        # Calcul du capital
        data['capital'] = initial_capital * data['cumulative_strategy_returns']
        
        # Calcul des commissions
        data['trade'] = data['position'].diff().fillna(0) != 0
        data['commission'] = data['trade'] * data['capital'] * commission
        data['capital_after_commission'] = data['capital'] - data['commission'].cumsum()
        
        # Calcul des métriques de performance
        total_return = data['cumulative_strategy_returns'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(data)) - 1
        
        # Calcul du drawdown
        data['drawdown'] = 1 - data['capital'] / data['capital'].cummax()
        max_drawdown = data['drawdown'].max()
        
        # Calcul du ratio de Sharpe (supposant un taux sans risque de 0%)
        sharpe_ratio = np.sqrt(252) * data['strategy_returns'].mean() / data['strategy_returns'].std()
        
        # Calcul du nombre de trades
        trades = data['trade'].sum()
        winning_trades = ((data['strategy_returns'] > 0) & data['trade']).sum()
        losing_trades = ((data['strategy_returns'] < 0) & data['trade']).sum()
        win_rate = winning_trades / trades if trades > 0 else 0
        
        # Résultats du backtest
        results = {
            'crypto_pair': crypto_pair,
            'strategy': strategy_func.__name__,
            'parameters': parameters,
            'initial_capital': initial_capital,
            'final_capital': data['capital_after_commission'].iloc[-1],
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'commission_paid': data['commission'].sum(),
            'test_period': {
                'start': data.index[0].strftime('%Y-%m-%d'),
                'end': data.index[-1].strftime('%Y-%m-%d'),
                'days': (data.index[-1] - data.index[0]).days
            }
        }
        
        # Stockage des résultats
        self.results[f"{crypto_pair}_{strategy_func.__name__}"] = {
            'results': results,
            'data': data
        }
        
        self.logger.info(f"Backtest terminé. Rendement total: {total_return:.2%}, Ratio de Sharpe: {sharpe_ratio:.2f}")
        
        # Sauvegarde des résultats
        self._save_results(results, data, crypto_pair, strategy_func.__name__)
        
        return results
    
    def _save_results(self, results, data, crypto_pair, strategy_name):
        """
        Sauvegarde les résultats du backtest.
        
        Args:
            results (dict): Résultats du backtest
            data (pd.DataFrame): Données avec les signaux et les rendements
            crypto_pair (str): Paire de crypto-monnaies
            strategy_name (str): Nom de la stratégie
        """
        # Création du répertoire pour les résultats
        result_dir = os.path.join(self.results_dir, f"{crypto_pair}_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(result_dir, exist_ok=True)
        
        # Sauvegarde des résultats en JSON
        with open(os.path.join(result_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        # Sauvegarde des données en CSV
        data.to_csv(os.path.join(result_dir, 'backtest_data.csv'))
        
        # Génération des graphiques
        self._generate_performance_charts(data, result_dir, crypto_pair, strategy_name)
        
        self.logger.info(f"Résultats sauvegardés dans {result_dir}")
    
    def _generate_performance_charts(self, data, result_dir, crypto_pair, strategy_name):
        """
        Génère des graphiques de performance pour le backtest.
        
        Args:
            data (pd.DataFrame): Données avec les signaux et les rendements
            result_dir (str): Répertoire pour sauvegarder les graphiques
            crypto_pair (str): Paire de crypto-monnaies
            strategy_name (str): Nom de la stratégie
        """
        # Graphique des rendements cumulés
        plt.figure(figsize=(12, 6))
        plt.plot(data['cumulative_returns'], label='Buy & Hold')
        plt.plot(data['cumulative_strategy_returns'], label='Stratégie')
        plt.title(f'Rendements cumulés pour {crypto_pair} avec {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Rendement cumulé')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'cumulative_returns.png'))
        
        # Graphique du capital
        plt.figure(figsize=(12, 6))
        plt.plot(data['capital_after_commission'])
        plt.title(f'Évolution du capital pour {crypto_pair} avec {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'capital.png'))
        
        # Graphique du drawdown
        plt.figure(figsize=(12, 6))
        plt.plot(data['drawdown'])
        plt.title(f'Drawdown pour {crypto_pair} avec {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'drawdown.png'))
        
        # Graphique des signaux sur le prix
        plt.figure(figsize=(12, 6))
        plt.plot(data['close'], label='Prix')
        plt.scatter(data[data['signal'] == 1].index, data['close'][data['signal'] == 1], 
                   marker='^', color='g', label='Achat')
        plt.scatter(data[data['signal'] == -1].index, data['close'][data['signal'] == -1], 
                   marker='v', color='r', label='Vente')
        plt.title(f'Signaux de trading pour {crypto_pair} avec {strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Prix')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(result_dir, 'signals.png'))
        
        plt.close('all')
    
    def compare_strategies(self, crypto_pair, strategies_results):
        """
        Compare les performances de différentes stratégies.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            strategies_results (list): Liste des résultats de stratégies à comparer
            
        Returns:
            pd.DataFrame: DataFrame comparant les performances des stratégies
        """
        if not strategies_results:
            self.logger.error("Aucun résultat de stratégie à comparer")
            return None
        
        # Création d'un DataFrame pour la comparaison
        comparison = pd.DataFrame()
        
        for result in strategies_results:
            strategy_name = result['strategy']
            comparison[strategy_name] = [
                result['total_return'],
                result['annualized_return'],
                result['max_drawdown'],
                result['sharpe_ratio'],
                result['trades'],
                result['win_rate']
            ]
        
        comparison.index = ['Total Return', 'Annualized Return', 'Max Drawdown', 'Sharpe Ratio', 'Trades', 'Win Rate']
        
        # Sauvegarde de la comparaison
        comparison_file = os.path.join(self.results_dir, f"{crypto_pair}_strategies_comparison.csv")
        comparison.to_csv(comparison_file)
        
        # Génération d'un graphique de comparaison
        plt.figure(figsize=(12, 8))
        comparison.loc['Total Return'].plot(kind='bar')
        plt.title(f'Comparaison des rendements totaux pour {crypto_pair}')
        plt.ylabel('Rendement total')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"{crypto_pair}_returns_comparison.png"))
        
        plt.figure(figsize=(12, 8))
        comparison.loc['Sharpe Ratio'].plot(kind='bar')
        plt.title(f'Comparaison des ratios de Sharpe pour {crypto_pair}')
        plt.ylabel('Ratio de Sharpe')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, f"{crypto_pair}_sharpe_comparison.png"))
        
        plt.close('all')
        
        self.logger.info(f"Comparaison des stratégies sauvegardée dans {comparison_file}")
        
        return comparison
    
    def optimize_strategy(self, strategy_func, crypto_pair, param_grid, initial_capital=10000.0, commission=0.001):
        """
        Optimise les paramètres d'une stratégie par grid search.
        
        Args:
            strategy_func (callable): Fonction de stratégie à optimiser
            crypto_pair (str): Paire de crypto-monnaies
            param_grid (dict): Grille de paramètres à tester
            initial_capital (float): Capital initial
            commission (float): Taux de commission
            
        Returns:
            dict: Meilleurs paramètres et résultats
        """
        if crypto_pair not in self.data:
            self.logger.error(f"Données non disponibles pour {crypto_pair}. Veuillez charger les données d'abord.")
            return None
        
        self.logger.info(f"Optimisation de la stratégie {strategy_func.__name__} pour {crypto_pair}")
        
        # Génération de toutes les combinaisons de paramètres
        import itertools
        param_keys = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Exécution des backtests pour chaque combinaison de paramètres
        results = []
        best_sharpe = -float('inf')
        best_params = None
        best_result = None
        
        for i, combination in enumerate(param_combinations):
            params = {param_keys[j]: combination[j] for j in range(len(param_keys))}
            self.logger.info(f"Test de la combinaison {i+1}/{len(param_combinations)}: {params}")
            
            result = self.run_backtest(strategy_func, crypto_pair, params, initial_capital, commission)
            results.append(result)
            
            # Mise à jour des meilleurs paramètres
            if result['sharpe_ratio'] > best_sharpe:
                best_sharpe = result['sharpe_ratio']
                best_params = params
                best_result = result
        
        # Sauvegarde des résultats d'optimisation
        optimization_results = {
            'crypto_pair': crypto_pair,
            'strategy': strategy_func.__name__,
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'best_return': best_result['total_return'],
            'all_results': results
        }
        
        optimization_file = os.path.join(self.results_dir, f"{crypto_pair}_{strategy_func.__name__}_optimization.json")
        with open(optimization_file, 'w') as f:
            json.dump(optimization_results, f, indent=4, default=str)
        
        self.logger.info(f"Optimisation terminée. Meilleurs paramètres: {best_params}, Sharpe: {best_sharpe:.2f}")
        
        # Génération d'un graphique des résultats d'optimisation
        self._plot_optimization_results(results, param_keys, optimization_file)
        
        return optimization_results
    
    def _plot_optimization_results(self, results, param_keys, optimization_file):
        """
        Génère des graphiques pour visualiser les résultats d'optimisation.
        
        Args:
            results (list): Liste des résultats de backtest
            param_keys (list): Liste des noms de paramètres
            optimization_file (str): Chemin du fichier d'optimisation
        """
        # Création d'un DataFrame avec les résultats
        df_results = pd.DataFrame()
        
        for result in results:
            row = {}
            for key in param_keys:
                row[key] = result['parameters'][key]
            
            row['total_return'] = result['total_return']
            row['sharpe_ratio'] = result['sharpe_ratio']
            row['max_drawdown'] = result['max_drawdown']
            
            df_results = df_results.append(row, ignore_index=True)
        
        # Sauvegarde des résultats en CSV
        csv_file = optimization_file.replace('.json', '.csv')
        df_results.to_csv(csv_file, index=False)
        
        # Si nous avons un ou deux paramètres, nous pouvons créer des graphiques plus informatifs
        if len(param_keys) == 1:
            plt.figure(figsize=(12, 6))
            plt.scatter(df_results[param_keys[0]], df_results['sharpe_ratio'])
            plt.title(f'Ratio de Sharpe en fonction de {param_keys[0]}')
            plt.xlabel(param_keys[0])
            plt.ylabel('Ratio de Sharpe')
            plt.grid(True)
            plt.savefig(optimization_file.replace('.json', '_sharpe.png'))
            
            plt.figure(figsize=(12, 6))
            plt.scatter(df_results[param_keys[0]], df_results['total_return'])
            plt.title(f'Rendement total en fonction de {param_keys[0]}')
            plt.xlabel(param_keys[0])
            plt.ylabel('Rendement total')
            plt.grid(True)
            plt.savefig(optimization_file.replace('.json', '_return.png'))
        
        elif len(param_keys) == 2:
            # Création d'une heatmap pour deux paramètres
            try:
                import seaborn as sns
                
                # Heatmap pour le ratio de Sharpe
                plt.figure(figsize=(12, 10))
                pivot_table = df_results.pivot_table(index=param_keys[0], columns=param_keys[1], values='sharpe_ratio')
                sns.heatmap(pivot_table, annot=True, cmap='viridis')
                plt.title(f'Ratio de Sharpe en fonction de {param_keys[0]} et {param_keys[1]}')
                plt.savefig(optimization_file.replace('.json', '_sharpe_heatmap.png'))
                
                # Heatmap pour le rendement total
                plt.figure(figsize=(12, 10))
                pivot_table = df_results.pivot_table(index=param_keys[0], columns=param_keys[1], values='total_return')
                sns.heatmap(pivot_table, annot=True, cmap='viridis')
                plt.title(f'Rendement total en fonction de {param_keys[0]} et {param_keys[1]}')
                plt.savefig(optimization_file.replace('.json', '_return_heatmap.png'))
            
            except ImportError:
                self.logger.warning("Seaborn non disponible. Les heatmaps ne seront pas générées.")
        
        plt.close('all')
    
    def monte_carlo_simulation(self, crypto_pair, strategy_func, parameters, num_simulations=1000, 
                              initial_capital=10000.0, commission=0.001):
        """
        Effectue une simulation Monte Carlo pour évaluer la robustesse d'une stratégie.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            strategy_func (callable): Fonction de stratégie à tester
            parameters (dict): Paramètres de la stratégie
            num_simulations (int): Nombre de simulations
            initial_capital (float): Capital initial
            commission (float): Taux de commission
            
        Returns:
            dict: Résultats de la simulation Monte Carlo
        """
        if crypto_pair not in self.data:
            self.logger.error(f"Données non disponibles pour {crypto_pair}. Veuillez charger les données d'abord.")
            return None
        
        data = self.data[crypto_pair].copy()
        
        self.logger.info(f"Exécution de la simulation Monte Carlo pour {crypto_pair} avec {num_simulations} simulations")
        
        # Génération des signaux
        signals = strategy_func(data, parameters)
        data['signal'] = signals
        
        # Calcul des rendements de la stratégie
        data['returns'] = data['close'].pct_change()
        data['strategy_returns'] = data['signal'].shift(1) * data['returns']
        
        # Suppression des valeurs NaN
        strategy_returns = data['strategy_returns'].dropna()
        
        # Simulation Monte Carlo
        simulation_results = []
        for i in range(num_simulations):
            # Échantillonnage aléatoire avec remplacement des rendements de la stratégie
            sampled_returns = np.random.choice(strategy_returns, size=len(strategy_returns), replace=True)
            
            # Calcul du capital cumulé
            cumulative_returns = (1 + sampled_returns).cumprod()
            final_capital = initial_capital * cumulative_returns[-1]
            
            # Calcul du drawdown
            drawdown = 1 - cumulative_returns / np.maximum.accumulate(cumulative_returns)
            max_drawdown = drawdown.max()
            
            simulation_results.append({
                'final_capital': final_capital,
                'total_return': cumulative_returns[-1] - 1,
                'max_drawdown': max_drawdown
            })
        
        # Analyse des résultats
        final_capitals = [result['final_capital'] for result in simulation_results]
        total_returns = [result['total_return'] for result in simulation_results]
        max_drawdowns = [result['max_drawdown'] for result in simulation_results]
        
        # Calcul des statistiques
        mean_final_capital = np.mean(final_capitals)
        median_final_capital = np.median(final_capitals)
        std_final_capital = np.std(final_capitals)
        
        mean_return = np.mean(total_returns)
        median_return = np.median(total_returns)
        std_return = np.std(total_returns)
        
        mean_max_drawdown = np.mean(max_drawdowns)
        median_max_drawdown = np.median(max_drawdowns)
        worst_max_drawdown = np.max(max_drawdowns)
        
        # Calcul des percentiles
        percentiles = [5, 25, 50, 75, 95]
        capital_percentiles = np.percentile(final_capitals, percentiles)
        return_percentiles = np.percentile(total_returns, percentiles)
        drawdown_percentiles = np.percentile(max_drawdowns, percentiles)
        
        # Résultats de la simulation
        monte_carlo_results = {
            'crypto_pair': crypto_pair,
            'strategy': strategy_func.__name__,
            'parameters': parameters,
            'num_simulations': num_simulations,
            'initial_capital': initial_capital,
            'capital_statistics': {
                'mean': mean_final_capital,
                'median': median_final_capital,
                'std': std_final_capital,
                'percentiles': {str(p): v for p, v in zip(percentiles, capital_percentiles)}
            },
            'return_statistics': {
                'mean': mean_return,
                'median': median_return,
                'std': std_return,
                'percentiles': {str(p): v for p, v in zip(percentiles, return_percentiles)}
            },
            'drawdown_statistics': {
                'mean': mean_max_drawdown,
                'median': median_max_drawdown,
                'worst': worst_max_drawdown,
                'percentiles': {str(p): v for p, v in zip(percentiles, drawdown_percentiles)}
            }
        }
        
        # Sauvegarde des résultats
        monte_carlo_file = os.path.join(self.results_dir, f"{crypto_pair}_{strategy_func.__name__}_monte_carlo.json")
        with open(monte_carlo_file, 'w') as f:
            json.dump(monte_carlo_results, f, indent=4, default=str)
        
        # Génération des graphiques
        self._plot_monte_carlo_results(final_capitals, total_returns, max_drawdowns, monte_carlo_file)
        
        self.logger.info(f"Simulation Monte Carlo terminée. Rendement moyen: {mean_return:.2%}, Pire drawdown: {worst_max_drawdown:.2%}")
        
        return monte_carlo_results
    
    def _plot_monte_carlo_results(self, final_capitals, total_returns, max_drawdowns, monte_carlo_file):
        """
        Génère des graphiques pour visualiser les résultats de la simulation Monte Carlo.
        
        Args:
            final_capitals (list): Liste des capitaux finaux
            total_returns (list): Liste des rendements totaux
            max_drawdowns (list): Liste des drawdowns maximaux
            monte_carlo_file (str): Chemin du fichier de résultats
        """
        # Histogramme des capitaux finaux
        plt.figure(figsize=(12, 6))
        plt.hist(final_capitals, bins=50, alpha=0.75)
        plt.axvline(np.median(final_capitals), color='r', linestyle='dashed', linewidth=2, label=f'Médiane: {np.median(final_capitals):.2f}')
        plt.title('Distribution des capitaux finaux')
        plt.xlabel('Capital final')
        plt.ylabel('Fréquence')