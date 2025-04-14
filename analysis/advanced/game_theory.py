import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional, Any
from scipy.optimize import minimize
from scipy.stats import norm
import networkx as nx
from sklearn.cluster import KMeans

class GameTheory:
    """
    Implementation of game theoretic models for cryptocurrency market analysis.
    Includes Nash equilibrium analysis, evolutionary game theory, auction theory,
    and strategic interaction modeling.
    """
    
    def __init__(self):
        """Initialize the GameTheory class."""
        pass
    
    def nash_equilibrium(self, payoff_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Find Nash equilibrium in a two-player game.
        
        Args:
            payoff_matrix (np.ndarray): 3D array where payoff_matrix[i,j] = [payoff1, payoff2]
            
        Returns:
            Dict: Dictionary with Nash equilibrium results
        """
        n_strategies = payoff_matrix.shape[0]
        results = {}
        
        # Find pure strategy Nash equilibria
        pure_equilibria = []
        for i in range(n_strategies):
            for j in range(n_strategies):
                # Check if strategy i is best response for player 1
                best_response1 = True
                for k in range(n_strategies):
                    if payoff_matrix[k,j,0] > payoff_matrix[i,j,0]:
                        best_response1 = False
                        break
                
                # Check if strategy j is best response for player 2
                best_response2 = True
                for k in range(n_strategies):
                    if payoff_matrix[i,k,1] > payoff_matrix[i,j,1]:
                        best_response2 = False
                        break
                
                if best_response1 and best_response2:
                    pure_equilibria.append((i, j))
        
        results['pure_equilibria'] = pure_equilibria
        
        # Find mixed strategy Nash equilibrium
        def expected_payoff(p, q):
            # Calculate expected payoff for player 1
            payoff1 = 0
            for i in range(n_strategies):
                for j in range(n_strategies):
                    payoff1 += p[i] * q[j] * payoff_matrix[i,j,0]
            
            # Calculate expected payoff for player 2
            payoff2 = 0
            for i in range(n_strategies):
                for j in range(n_strategies):
                    payoff2 += p[i] * q[j] * payoff_matrix[i,j,1]
            
            return payoff1, payoff2
        
        # Find mixed strategy equilibrium using optimization
        def objective(x):
            p = x[:n_strategies]
            q = x[n_strategies:]
            payoff1, payoff2 = expected_payoff(p, q)
            return -(payoff1 + payoff2)  # Maximize total payoff
        
        # Constraints: probabilities sum to 1
        cons = ({'type': 'eq', 'fun': lambda x: np.sum(x[:n_strategies]) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(x[n_strategies:]) - 1})
        
        # Bounds: probabilities between 0 and 1
        bounds = [(0, 1) for _ in range(2 * n_strategies)]
        
        # Initial guess
        x0 = np.ones(2 * n_strategies) / n_strategies
        
        # Optimize
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        if res.success:
            p_opt = res.x[:n_strategies]
            q_opt = res.x[n_strategies:]
            payoff1, payoff2 = expected_payoff(p_opt, q_opt)
            results['mixed_equilibrium'] = {
                'player1_strategy': p_opt,
                'player2_strategy': q_opt,
                'payoff1': payoff1,
                'payoff2': payoff2
            }
        
        return results
    
    def evolutionary_game(self, payoff_matrix: np.ndarray, 
                         initial_population: np.ndarray,
                         generations: int = 100,
                         mutation_rate: float = 0.01) -> Dict[str, Any]:
        """
        Simulate evolutionary game dynamics.
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix for the game
            initial_population (np.ndarray): Initial population distribution
            generations (int): Number of generations to simulate
            mutation_rate (float): Rate of mutation
            
        Returns:
            Dict: Dictionary with evolutionary game results
        """
        n_strategies = len(initial_population)
        population = initial_population.copy()
        history = np.zeros((generations, n_strategies))
        
        for gen in range(generations):
            # Calculate fitness
            fitness = np.zeros(n_strategies)
            for i in range(n_strategies):
                for j in range(n_strategies):
                    fitness[i] += population[j] * payoff_matrix[i,j]
            
            # Calculate average fitness
            avg_fitness = np.sum(population * fitness)
            
            # Update population (replicator dynamics)
            if avg_fitness > 0:
                population = population * fitness / avg_fitness
            
            # Apply mutation
            mutation = np.random.normal(0, mutation_rate, n_strategies)
            population = np.maximum(0, population + mutation)
            population = population / np.sum(population)  # Normalize
            
            # Record history
            history[gen] = population
        
        return {
            'final_population': population,
            'history': history,
            'generations': generations
        }
    
    def auction_theory(self, valuations: np.ndarray, 
                      reserve_price: float = 0.0,
                      auction_type: str = 'first_price') -> Dict[str, Any]:
        """
        Analyze auction outcomes using game theory.
        
        Args:
            valuations (np.ndarray): Array of bidder valuations
            reserve_price (float): Minimum acceptable price
            auction_type (str): Type of auction ('first_price', 'second_price', 'all_pay')
            
        Returns:
            Dict: Dictionary with auction analysis results
        """
        n_bidders = len(valuations)
        results = {}
        
        if auction_type == 'first_price':
            # First-price sealed-bid auction
            # Equilibrium strategy: bid (n-1)/n * valuation
            equilibrium_bids = [(n_bidders-1)/n_bidders * v for v in valuations]
            winning_bid = max(equilibrium_bids)
            winner = np.argmax(equilibrium_bids)
            
            results.update({
                'equilibrium_bids': equilibrium_bids,
                'winning_bid': winning_bid,
                'winner': winner,
                'revenue': winning_bid if winning_bid >= reserve_price else 0
            })
            
        elif auction_type == 'second_price':
            # Second-price sealed-bid auction
            # Equilibrium strategy: bid true valuation
            bids = valuations
            sorted_bids = np.sort(bids)
            winning_bid = sorted_bids[-2] if len(bids) > 1 else reserve_price
            winner = np.argmax(bids)
            
            results.update({
                'equilibrium_bids': bids,
                'winning_bid': winning_bid,
                'winner': winner,
                'revenue': winning_bid if winning_bid >= reserve_price else 0
            })
            
        elif auction_type == 'all_pay':
            # All-pay auction
            # Equilibrium strategy: bid (n-1)/n * valuation^n
            equilibrium_bids = [(n_bidders-1)/n_bidders * v**n_bidders for v in valuations]
            winner = np.argmax(valuations)
            
            results.update({
                'equilibrium_bids': equilibrium_bids,
                'total_payments': sum(equilibrium_bids),
                'winner': winner,
                'revenue': sum(equilibrium_bids)
            })
        
        return results
    
    def market_making_game(self, fundamental_value: float,
                          noise_traders: int = 100,
                          market_makers: int = 5,
                          rounds: int = 100) -> Dict[str, Any]:
        """
        Simulate a market making game with noise traders and market makers.
        
        Args:
            fundamental_value (float): True value of the asset
            noise_traders (int): Number of noise traders
            market_makers (int): Number of market makers
            rounds (int): Number of trading rounds
            
        Returns:
            Dict: Dictionary with market making game results
        """
        # Initialize market makers' inventories and quotes
        inventories = np.zeros(market_makers)
        bid_quotes = np.zeros((rounds, market_makers))
        ask_quotes = np.zeros((rounds, market_makers))
        prices = np.zeros(rounds)
        
        # Market maker parameters
        inventory_aversion = 0.1
        spread_competition = 0.02
        
        for t in range(rounds):
            # Market makers set quotes based on inventory and competition
            for i in range(market_makers):
                # Adjust quotes based on inventory
                inventory_effect = -inventory_aversion * inventories[i]
                
                # Set bid and ask quotes
                bid_quotes[t,i] = fundamental_value + inventory_effect - spread_competition/2
                ask_quotes[t,i] = fundamental_value + inventory_effect + spread_competition/2
            
            # Noise traders arrive and trade
            noise_demand = np.random.normal(0, 1, noise_traders)
            
            # Match trades with best quotes
            for demand in noise_demand:
                if demand > 0:  # Buy order
                    best_ask = np.min(ask_quotes[t])
                    best_ask_maker = np.argmin(ask_quotes[t])
                    if best_ask <= fundamental_value * (1 + 0.1):  # Price limit
                        inventories[best_ask_maker] -= 1
                        prices[t] = best_ask
                else:  # Sell order
                    best_bid = np.max(bid_quotes[t])
                    best_bid_maker = np.argmax(bid_quotes[t])
                    if best_bid >= fundamental_value * (1 - 0.1):  # Price limit
                        inventories[best_bid_maker] += 1
                        prices[t] = best_bid
        
        return {
            'prices': prices,
            'bid_quotes': bid_quotes,
            'ask_quotes': ask_quotes,
            'inventories': inventories,
            'spread': np.mean(ask_quotes - bid_quotes),
            'price_volatility': np.std(prices)
        }
    
    def coordination_game(self, payoff_matrix: np.ndarray,
                         initial_state: np.ndarray,
                         network: Optional[nx.Graph] = None,
                         rounds: int = 100) -> Dict[str, Any]:
        """
        Simulate coordination game on a network.
        
        Args:
            payoff_matrix (np.ndarray): Payoff matrix for coordination
            initial_state (np.ndarray): Initial strategy distribution
            network (nx.Graph, optional): Interaction network
            rounds (int): Number of rounds to simulate
            
        Returns:
            Dict: Dictionary with coordination game results
        """
        n_players = len(initial_state)
        
        # Create random network if none provided
        if network is None:
            network = nx.erdos_renyi_graph(n_players, 0.3)
        
        # Initialize states
        states = initial_state.copy()
        history = np.zeros((rounds, n_players))
        
        for t in range(rounds):
            # Players update strategies based on neighbors
            new_states = states.copy()
            
            for i in range(n_players):
                # Get neighbors
                neighbors = list(network.neighbors(i))
                if not neighbors:
                    continue
                
                # Calculate average payoff for each strategy
                avg_payoffs = np.zeros(2)
                for strategy in [0, 1]:
                    # Count neighbors playing each strategy
                    neighbor_states = states[neighbors]
                    n_same = np.sum(neighbor_states == strategy)
                    n_diff = len(neighbors) - n_same
                    
                    # Calculate expected payoff
                    avg_payoffs[strategy] = (n_same * payoff_matrix[strategy,strategy] +
                                           n_diff * payoff_matrix[strategy,1-strategy]) / len(neighbors)
                
                # Update strategy (best response)
                new_states[i] = np.argmax(avg_payoffs)
            
            states = new_states
            history[t] = states
        
        return {
            'final_states': states,
            'history': history,
            'convergence': np.all(states == states[0]),  # Check if all players chose same strategy
            'network': network
        }
    
    def plot_evolutionary_dynamics(self, results: Dict[str, Any], 
                                 title: str = "Evolutionary Game Dynamics") -> plt.Figure:
        """
        Plot evolutionary game dynamics.
        
        Args:
            results (Dict): Results from evolutionary_game
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        history = results['history']
        generations = results['generations']
        n_strategies = history.shape[1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(n_strategies):
            ax.plot(range(generations), history[:,i], label=f'Strategy {i+1}')
        
        ax.set_xlabel('Generation')
        ax.set_ylabel('Population Share')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_market_making(self, results: Dict[str, Any], 
                          title: str = "Market Making Game") -> plt.Figure:
        """
        Plot market making game results.
        
        Args:
            results (Dict): Results from market_making_game
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        prices = results['prices']
        bid_quotes = results['bid_quotes']
        ask_quotes = results['ask_quotes']
        rounds = len(prices)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot prices and quotes
        ax1.plot(range(rounds), prices, 'k-', label='Transaction Price')
        ax1.plot(range(rounds), np.mean(bid_quotes, axis=1), 'b--', label='Average Bid')
        ax1.plot(range(rounds), np.mean(ask_quotes, axis=1), 'r--', label='Average Ask')
        ax1.set_ylabel('Price')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot inventories
        for i in range(bid_quotes.shape[1]):
            ax2.plot(range(rounds), results['inventories'], 
                    label=f'Market Maker {i+1}')
        ax2.set_xlabel('Round')
        ax2.set_ylabel('Inventory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def analyze_market_structure(self, returns_df: pd.DataFrame,
                               threshold: float = 0.5) -> Dict[str, Any]:
        """
        Analyze market structure using game theoretic concepts.
        
        Args:
            returns_df (pd.DataFrame): DataFrame with asset returns
            threshold (float): Correlation threshold for network construction
            
        Returns:
            Dict: Dictionary with market structure analysis
        """
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Create network based on correlations
        G = nx.Graph()
        for i in range(len(corr_matrix)):
            G.add_node(i)
            for j in range(i+1, len(corr_matrix)):
                if abs(corr_matrix.iloc[i,j]) > threshold:
                    G.add_edge(i, j, weight=abs(corr_matrix.iloc[i,j]))
        
        # Calculate network metrics
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        clustering = nx.clustering(G)
        
        # Identify key players (market makers)
        key_players = sorted(degree_centrality.items(), 
                           key=lambda x: x[1], 
                           reverse=True)[:5]
        
        # Analyze market power
        market_power = {}
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            if neighbors:
                # Market power based on degree and clustering
                market_power[node] = (degree_centrality[node] * 
                                    (1 - clustering[node]))
        
        return {
            'network': G,
            'degree_centrality': degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'clustering': clustering,
            'key_players': key_players,
            'market_power': market_power
        } 