import pandas as pd
import numpy as np
import logging
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

class RiskManager:
    """
    Module de gestion des risques avancée pour le système de trading.
    Implémente Value-at-Risk (VaR), Conditional Value-at-Risk (CVaR),
    surveillance de liquidité, et contrôle dynamique des expositions.
    """
    
    def __init__(self, confidence_level=0.95, var_window=252, max_drawdown_limit=0.2,
                 max_position_size=0.1, max_leverage=2.0, rebalance_threshold=0.05):
        """
        Initialise le gestionnaire de risques.
        
        Args:
            confidence_level (float): Niveau de confiance pour VaR et CVaR (0.95 = 95%)
            var_window (int): Fenêtre temporelle pour le calcul de VaR (jours)
            max_drawdown_limit (float): Limite maximale de drawdown autorisée
            max_position_size (float): Taille maximale d'une position en % du portefeuille
            max_leverage (float): Effet de levier maximal autorisé
            rebalance_threshold (float): Seuil de déséquilibre pour le rééquilibrage
        """
        self.confidence_level = confidence_level
        self.var_window = var_window
        self.max_drawdown_limit = max_drawdown_limit
        self.max_position_size = max_position_size
        self.max_leverage = max_leverage
        self.rebalance_threshold = rebalance_threshold
        
        self.portfolio_history = []
        self.risk_metrics_history = {}
        self.position_limits = {}
        self.exposure_history = {}
        self.liquidity_metrics = {}
        self.transaction_costs = {}
        
        self.log_dir = os.path.join(os.getcwd(), 'logs')
        self.results_dir = os.path.join(os.getcwd(), 'results')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Configuration du logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'risk_manager.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('RiskManager')
        self.logger.info("Module de gestion des risques initialisé")
    
    def calculate_var(self, returns, confidence_level=None):
        """
        Calcule la Value-at-Risk (VaR) pour une série de rendements.
        
        Args:
            returns (pd.Series): Série de rendements historiques
            confidence_level (float): Niveau de confiance (si None, utilise la valeur par défaut)
            
        Returns:
            float: Value-at-Risk au niveau de confiance spécifié
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Vérifier s'il y a suffisamment de données
        if len(returns) < 30:
            self.logger.warning("Données insuffisantes pour calculer la VaR de manière fiable")
            return None
        
        # Calcul de la VaR historique
        var = np.percentile(returns, 100 * (1 - confidence_level))
        
        self.logger.info(f"VaR calculée: {var:.4f} au niveau de confiance {confidence_level:.2f}")
        return var
    
    def calculate_cvar(self, returns, confidence_level=None):
        """
        Calcule la Conditional Value-at-Risk (CVaR) pour une série de rendements.
        
        Args:
            returns (pd.Series): Série de rendements historiques
            confidence_level (float): Niveau de confiance (si None, utilise la valeur par défaut)
            
        Returns:
            float: Conditional Value-at-Risk au niveau de confiance spécifié
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Vérifier s'il y a suffisamment de données
        if len(returns) < 30:
            self.logger.warning("Données insuffisantes pour calculer la CVaR de manière fiable")
            return None
        
        # Calcul de la VaR
        var = self.calculate_var(returns, confidence_level)
        
        # Calcul de la CVaR (moyenne des rendements inférieurs à la VaR)
        cvar = returns[returns <= var].mean()
        
        self.logger.info(f"CVaR calculée: {cvar:.4f} au niveau de confiance {confidence_level:.2f}")
        return cvar
    
    def calculate_portfolio_var(self, weights, cov_matrix, expected_returns, confidence_level=None):
        """
        Calcule la VaR d'un portefeuille.
        
        Args:
            weights (np.array): Poids des actifs dans le portefeuille
            cov_matrix (pd.DataFrame): Matrice de covariance des rendements
            expected_returns (np.array): Rendements attendus des actifs
            confidence_level (float): Niveau de confiance (si None, utilise la valeur par défaut)
            
        Returns:
            float: VaR du portefeuille
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Calcul du rendement attendu du portefeuille
        portfolio_return = np.sum(weights * expected_returns)
        
        # Calcul de la volatilité du portefeuille
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Calcul de la VaR paramétrique
        z_score = stats.norm.ppf(1 - confidence_level)
        var = -(portfolio_return + z_score * portfolio_volatility)
        
        self.logger.info(f"VaR du portefeuille calculée: {var:.4f}")
        return var
    
    def monitor_liquidity(self, crypto_pair, order_book_data, position_size):
        """
        Surveille la liquidité du marché et évalue l'impact potentiel des transactions.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            order_book_data (dict): Données du carnet d'ordres
            position_size (float): Taille de la position à prendre/liquider
            
        Returns:
            dict: Métriques de liquidité et impact estimé
        """
        # Extraire les données du carnet d'ordres
        bids = order_book_data.get('bids', [])
        asks = order_book_data.get('asks', [])
        
        if not bids or not asks:
            self.logger.warning(f"Données de carnet d'ordres insuffisantes pour {crypto_pair}")
            return None
        
        # Calculer la profondeur du marché
        bid_depth = sum(float(bid[0]) * float(bid[1]) for bid in bids)
        ask_depth = sum(float(ask[0]) * float(ask[1]) for ask in asks)
        total_depth = bid_depth + ask_depth
        
        # Calculer le spread
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = (best_ask - best_bid) / best_bid
        
        # Estimer l'impact de marché
        impact_ratio = position_size / total_depth
        estimated_slippage = spread * (1 + impact_ratio * 10)  # Formule simplifiée
        
        # Métriques de liquidité
        liquidity_metrics = {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': total_depth,
            'spread': spread,
            'impact_ratio': impact_ratio,
            'estimated_slippage': estimated_slippage,
            'timestamp': datetime.now()
        }
        
        # Stocker les métriques
        self.liquidity_metrics[crypto_pair] = liquidity_metrics
        
        self.logger.info(f"Métriques de liquidité calculées pour {crypto_pair}: spread={spread:.6f}, impact={impact_ratio:.6f}")
        return liquidity_metrics
    
    def adjust_position_size(self, crypto_pair, base_position_size, market_data, risk_metrics):
        """
        Ajuste la taille de position en fonction des métriques de risque et de liquidité.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            base_position_size (float): Taille de position de base
            market_data (dict): Données de marché actuelles
            risk_metrics (dict): Métriques de risque calculées
            
        Returns:
            float: Taille de position ajustée
        """
        # Extraire les métriques de risque
        var = risk_metrics.get('var', 0.05)
        volatility = risk_metrics.get('volatility', 0.02)
        liquidity_impact = risk_metrics.get('liquidity_impact', 0.01)
        current_exposure = risk_metrics.get('current_exposure', 0)
        
        # Facteurs d'ajustement
        var_factor = 1 - (var / 0.1)  # Réduire la position si VaR élevée
        vol_factor = 1 - (volatility / 0.05)  # Réduire la position si volatilité élevée
        liquidity_factor = 1 - (liquidity_impact / 0.02)  # Réduire la position si impact de liquidité élevé
        
        # Limiter les facteurs entre 0.2 et 1.5
        var_factor = max(0.2, min(1.5, var_factor))
        vol_factor = max(0.2, min(1.5, vol_factor))
        liquidity_factor = max(0.2, min(1.5, liquidity_factor))
        
        # Calculer la taille de position ajustée
        adjusted_size = base_position_size * var_factor * vol_factor * liquidity_factor
        
        # Limiter la taille de position au maximum autorisé
        max_size = self.max_position_size
        adjusted_size = min(adjusted_size, max_size)
        
        # Vérifier l'exposition totale
        if current_exposure + adjusted_size > self.max_leverage:
            adjusted_size = max(0, self.max_leverage - current_exposure)
        
        self.logger.info(f"Taille de position ajustée pour {crypto_pair}: {adjusted_size:.4f} (base: {base_position_size:.4f})")
        return adjusted_size
    
    def calculate_optimal_portfolio(self, expected_returns, cov_matrix, risk_free_rate=0.01):
        """
        Calcule le portefeuille optimal selon la théorie moderne du portefeuille.
        
        Args:
            expected_returns (np.array): Rendements attendus des actifs
            cov_matrix (pd.DataFrame): Matrice de covariance des rendements
            risk_free_rate (float): Taux sans risque
            
        Returns:
            dict: Poids optimaux et métriques du portefeuille
        """
        num_assets = len(expected_returns)
        
        # Fonction à minimiser (inverse du ratio de Sharpe)
        def negative_sharpe_ratio(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            return -sharpe_ratio
        
        # Contraintes: somme des poids = 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Limites: pas de vente à découvert (poids entre 0 et 1)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Poids initiaux égaux
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimisation
        result = minimize(negative_sharpe_ratio, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        # Extraire les poids optimaux
        optimal_weights = result['x']
        
        # Calculer les métriques du portefeuille optimal
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        # Résultats
        optimal_portfolio = {
            'weights': optimal_weights,
            'return': portfolio_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio
        }
        
        self.logger.info(f"Portefeuille optimal calculé: rendement={portfolio_return:.4f}, volatilité={portfolio_volatility:.4f}, Sharpe={sharpe_ratio:.4f}")
        return optimal_portfolio
    
    def calculate_transaction_costs(self, crypto_pair, order_size, order_type, market_data):
        """
        Calcule les coûts de transaction estimés pour un ordre.
        
        Args:
            crypto_pair (str): Paire de crypto-monnaies
            order_size (float): Taille de l'ordre
            order_type (str): Type d'ordre ('market' ou 'limit')
            market_data (dict): Données de marché actuelles
            
        Returns:
            dict: Coûts de transaction estimés
        """
        # Extraire les données de marché pertinentes
        current_price = market_data.get('price', 0)
        spread = market_data.get('spread', 0.001)
        
        # Frais de base (dépend de la plateforme)
        base_fee_rate = 0.001  # 0.1% (typique pour Binance)
        base_fee = order_size * current_price * base_fee_rate
        
        # Coût de slippage (dépend du type d'ordre et de la liquidité)
        slippage = 0
        if order_type == 'market':
            # Pour les ordres au marché, le slippage est plus élevé
            liquidity_impact = market_data.get('liquidity_impact', 0.001)
            slippage = order_size * current_price * (spread + liquidity_impact)
        else:
            # Pour les ordres limites, le slippage est généralement nul ou négatif
            slippage = 0
        
        # Coût total
        total_cost = base_fee + slippage
        cost_percentage = (total_cost / (order_size * current_price)) * 100
        
        # Résultats
        transaction_costs = {
            'base_fee': base_fee,
            'slippage': slippage,
            'total_cost': total_cost,
            'cost_percentage': cost_percentage,
            'timestamp': datetime.now()
        }
        
        # Stocker les coûts
        self.transaction_costs[crypto_pair] = transaction_costs
        
        self.logger.info(f"Coûts de transaction calculés pour {crypto_pair}: {cost_percentage:.4f}%")
        return transaction_costs
    
    def generate_risk_report(self, portfolio_data, positions, transactions, start_date, end_date):
        """
        Génère un rapport de risque complet pour la période spécifiée.
        
        Args:
            portfolio_data (pd.DataFrame): Données historiques du portefeuille
            positions (dict): Positions actuelles
            transactions (list): Historique des transactions
            start_date (datetime): Date de début de la période
            end_date (datetime): Date de fin de la période
            
        Returns:
            dict: Rapport de risque complet
        """
        # Filtrer les données pour la période spécifiée
        mask = (portfolio_data.index >= start_date) & (portfolio_data.index <= end_date)
        period_data = portfolio_data.loc[mask]
        
        if len(period_data) == 0:
            self.logger.warning("Aucune donnée disponible pour la période spécifiée")
            return None
        
        # Calculer les rendements quotidiens
        daily_returns = period_data['total_value'].pct_change().dropna()
        
        # Métriques de performance
        total_return = (period_data['total_value'].iloc[-1] / period_data['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(period_data)) - 1
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Calcul du drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # Calcul de VaR et CVaR
        var_95 = self.calculate_var(daily_returns, 0.95)
        cvar_95 = self.calculate_cvar(daily_returns, 0.95)
        var_99 = self.calculate_var(daily_returns, 0.99)
        cvar_99 = self.calculate_cvar(daily_returns, 0.99)
        
        # Exposition par actif
        exposure_by_asset = {}
        for asset, position in positions.items():
            exposure = position['quantity'] * position['current_price']
            exposure_by_asset[asset] = exposure
        
        # Statistiques des transactions
        num_transactions = len([t for t in transactions if start_date <= t['timestamp'] <= end_date])
        transaction_volume = sum([t['quantity'] * t['price'] for t in transactions if start_date <= t['timestamp'] <= end_date])
        transaction_costs = sum([t['fee'] for t in transactions if start_date <= t['timestamp'] <= end_date])
        
        # Rapport complet
        risk_report = {
            'period': {
                'start_date': start_date,
                'end_date': end_date,
                'days': len(period_data)
            },
            'performance': {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            },
            'risk_metrics': {
                'var_95': var_95,
                'cvar_95': cvar_95,
                'var_99': var_99,
                'cvar_99': cvar_99
            },
            'exposure': exposure_by_asset,
            'transactions': {
                'count': num_transactions,
                'volume': transaction_volume,
                'costs': transaction_costs,
                'cost_percentage': transaction_costs / transaction_volume if transaction_volume > 0 else 0
            }
        }
        
        # Enregistrer le rapport
        report_file = os.path.join(self.results_dir, f"risk_report_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json")
        with open(report_file, 'w') as f:
            import json
            json.dump(risk_report, f, indent=4, default=str)
        
        self.logger.info(f"Rapport de risque généré pour la période du {start_date} au {end_date}")
        return risk_report
    
    def plot_risk_metrics(self, portfolio_data, risk_metrics, output_file=None):
        """
        Génère des graphiques des métriques de risque.
        
        Args:
            portfolio_data (pd.DataFrame): Données historiques du portefeuille
            risk_metrics (dict): Métriques de risque calculées
            output_file (str): Chemin du fichier de sortie pour le graphique
            
        Returns:
            bool: True si le graphique a été généré avec succès, False sinon
        """
        try:
            # Créer une figure avec plusieurs sous-graphiques
            fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
            
            # Graphique 1: Valeur du portefeuille et drawdown
            axs[0].plot(portfolio_data.index, portfolio_data['total_value'], label='Valeur du portefeuille')
            axs[0].set_title('Évolution de la valeur du portefeuille')
            axs[0].set_ylabel('Valeur')
            axs[0].legend()
            axs[0].grid(True)
            
            # Graphique 2: VaR et CVaR
            if 'var_95' in risk_metrics and 'var_95' in portfolio_data.columns:
                axs[1].plot(portfolio_data.index, portfolio_data['var_95'], 'r-', label='VaR (95%)')
                axs[1].plot(portfolio_data.index, portfolio_data['cvar_95'], 'b-', label='CVaR (95%)')
                axs[1].set_title('Value-at-Risk et Conditional Value-at-Risk')
                axs[1].set_ylabel('Valeur')
                axs[1].legend()
                axs[1].grid(True)
            
            # Graphique 3: Exposition par actif
            if 'exposure' in portfolio_data.columns:
                exposure_data = portfolio_data[['exposure']].unstack()
                exposure_data.plot(kind='area', stacked=True, ax=axs[2])
                axs[2].set_title('Exposition par actif')
                axs[2].set_ylabel('Exposition')
                axs[2].set_xlabel('Date')
                axs[2].legend()
                axs[2].grid(True)
            
            # Ajuster la mise en page
            plt.tight_layout()
            
            # Enregistrer le graphique si un fichier de sortie est spécifié
            if output_file:
                plt.savefig(output_file)
                self.logger.info(f"Graphique des métriques de risque enregistré dans {output_file}")
            
            plt.close(fig)
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du graphique: {e}")
            return False