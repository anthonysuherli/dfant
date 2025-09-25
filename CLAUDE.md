# NBA DFS Optimization Model Methodology

## Problem Domain
NBA Daily Fantasy Sports is a constrained optimization problem where players select 8 athletes (1 PG, 1 SG, 1 SF, 1 PF, 1 C, 1 G, 1 F, 1 UTIL) within a $50,000 salary cap to maximize fantasy points. The market exhibits extreme concentration with 1.3% of players capturing 91% of profits.

## Mathematical Foundations

### Kelly Criterion Adaptations
- Standard Kelly modified for multi-outcome distributions
- Half-Kelly sizing for DFS: f* = 0.5 × (bp - q) / b
- Portfolio allocation: 80% cash games, 20% tournaments
- Maximum 10% daily exposure

### Alternative Frameworks
- **Modern Portfolio Theory**: Optimize E(R) - λσ² for contest entries as assets
- **Rank-Dependent Expected Utility**: U = Σw(p_i)u(x_i) for skewed payouts
- **Shannon Entropy**: Maximize H = -Σp_i log(p_i) for diversification

## Machine Learning Architecture

### Performance Benchmarks
```
Baseline (season averages): 35-40% MAPE
Linear models:             32-35% MAPE
Random Forest:             30-32% MAPE
XGBoost:                   28-29% MAPE ⭐
Ensemble methods:          28.90-29.50% MAPE
```

### Optimal Model Structure
**Ensemble Approach**: 14 models with voting meta-learner
- 3 XGBoost variants (different feature sets)
- 2 LightGBM models
- 3 Neural networks (5-layer, ReLU activation: [256,128,64,32,16])
- 2 LSTM models (temporal patterns)
- 4 specialized models (position/team specific)

### Neural Network Configuration
- Input: 100-150 features
- Regularization: 0.3 dropout, 0.001 L2, batch normalization
- Architecture: 5-layer with ReLU activations throughout

## Feature Engineering Pipeline (Priority Order)

1. **Advanced Basketball Metrics** (1.7-1.9% MAPE improvement)
   - PER (Player Efficiency Rating)
   - True Shooting %: Points / (2 × (FGA + 0.44 × FTA))
   - Usage Rate calculation with team context

2. **Moving Averages** (optimal windows)
   - Consistent players: 10-15 games
   - Inconsistent players: 20-30 games
   - Exponential decay: w_i = α × (1-α)^i where α = 0.1-0.3

3. **Contextual Factors**
   - Home/away encoding
   - Rest days (0,1,2,3+)
   - Back-to-back games
   - Travel distance

4. **Injury Status**
   - Probability-based encoding (0-1 scale)
   - Teammate injury impact modeling

5. **Seasonal Momentum**
   - Win streaks, performance trends, clutch metrics

## Contest Strategy

### Cash Games (4x-6x Rule)
```
Target:      E[lineup_score] ≥ 5x average salary multiplier
Constraint:  P(lineup_score ≥ 4x) ≥ 0.85
Stretch:     P(lineup_score ≥ 6x) ≥ 0.15
```
- High floor players: σ/μ < 0.25
- Win rate requirement: >55% to overcome rake
- Risk management priority

### Tournaments (Leverage Zone Strategy)
- Target ownership: 10-30% (leverage zone)
- Avoid >40% ownership (chalk) and <5% ownership (too contrarian)
- Field differentiation: 2-4x exposure on leverage plays

## Validated Performance Results

### Academic Validation
- **Mlčoch (2024)**: €2,439 profit on €7,133 investment = 34.2% ROI over 8 weeks
- **Papageorgiou et al. (2024)**: Top 18.4% performance among 11,764 competitors
- Statistical significance: p < 0.01 vs random selection

### Professional Benchmarks
- Sustained ROI: 10-15% for systematic players
- Entry volume: 50-500 lineups per slate
- Bankroll requirements: 200-500x average daily exposure
- Maximum drawdown tolerance: 20%

## Technical Implementation


### Core Technology Stack
1. **XGBoost** with Bayesian hyperparameter optimization
2. **Ensemble meta-learning** for model combination
3. **Mixed-integer programming** for constraint optimization
4. **Real-time data pipeline** (30-second injury polling)

### Critical Success Factors
- Individual player modeling (vs generalized approaches)
- Professional-grade optimization software with correlation modeling
- Disciplined fractional Kelly bankroll management
- Real-time injury monitoring and late swap capabilities
- Ownership projection modeling

### Data Pipeline Requirements
- Real-time injury monitoring (API polling every 30 seconds)
- Vegas line movement tracking
- Ownership projections (pre-game and live updates)
- Automated late swap capability

## Future Directions

### Advanced Architectures
- **Transformer models**: Attention mechanisms for teammate chemistry
- **Graph Neural Networks**: Players as nodes, interactions as edges
- **Real-time adaptation**: Game flow multipliers updating every 2-3 minutes

### Key Implementation Insight
Treat DFS as an applied machine learning and optimization problem rather than sports prediction. Success requires mathematical rigor in bankroll management, predictive accuracy through ensemble modeling, and systematic execution with professional-grade technology infrastructure.

## Risk Management Framework
- Maximum 10% daily exposure
- Half-Kelly position sizing
- 200-500x bankroll depth requirement
- Performance tracking across multiple time horizons
- Professional players experience six-figure downswings despite positive expected value