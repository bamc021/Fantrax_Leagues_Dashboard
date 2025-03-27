# Fantrax Leagues Dashboard

A Streamlit-based dashboard for visualizing MLB and MiLB statistics for your Fantrax fantasy baseball leagues.

## Overview

Fantrax Leagues Dashboard provides real-time statistics visualization for fantasy baseball managers. The app pulls data from Fangraphs and your Fantrax leagues, combines them, and creates interactive dashboards to analyze player performance.

Key features:
- Team-level statistics for both hitting and pitching
- Individual MLB player statistics for active roster players
- Minor league statistics for prospects in your league
- Customizable visualizations with scatter plots
- Identification of missing player IDs

## Installation

### Prerequisites
- Python 3.8+
- Pip package manager

## Usage

1. **Select Your League**: Use the dropdown menu in the sidebar to select your Fantrax league.

2. **League Data Tab**: View aggregate team statistics for both hitting and pitching. Choose any metrics for X and Y axes to create custom scatter plots.

3. **MLB Stats Tab**: Analyze individual MLB player statistics for your rostered players. The scatter plots allow you to identify outliers and compare players across teams.

4. **MiLB Stats Tab**: Track the performance of minor league players in your dynasty leagues with customizable scatter plots.

5. **Troubleshoot Tab**: Identify players with missing IDs in the Player ID Key file for manual mapping.

6. **Refresh Data**: Click the "Refresh" button in the sidebar when you want to reload the latest data.

## Data Sources

- MLB statistics are pulled from Fangraphs API
- Minor league statistics are pulled from Fangraphs API
- Player rosters are fetched from Fantrax API
- Player ID mapping uses the Player ID Key system

## Troubleshooting

- If you see warnings about missing Fantrax IDs, check the Troubleshoot tab for a list of players that need to be added to the ID Key.
- Ensure your Fantrax account has access to the leagues you're trying to view.
- Check your internet connection if data fails to load.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the app framework
- [Fangraphs](https://www.fangraphs.com/) for the MLB and MiLB statistics
- [Fantrax](https://www.fantrax.com/) for the fantasy baseball platform 