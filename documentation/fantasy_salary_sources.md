# Fantasy Salary Data Sources

## Overview
The official NBA API does **not** provide fantasy-specific data like player salaries. However, there are several third-party APIs and resources that do.

## Available Options

### 1. SportsDataIO (Recommended for Personal Use)
- **Free Tier**: Last season's data (rosters, schedules, results)
- **Paid**: Current season data, fantasy feeds including:
  - Player salaries for FanDuel, DraftKings, Yahoo
  - Projections
  - Lineup optimizers
- **Pricing**: Free for personal use (last season), paid for current season
- **Website**: https://discoverylab.sportsdata.io/personal-use-apis/nba

### 2. Sportradar
- **Recently Added**: Player salary information
- **Endpoints**: Player Profile and Team Profile now include current base annual salaries
- **Note**: This is real NBA salaries, not DFS salaries
- **Website**: https://developer.sportradar.com

### 3. FantasyData.com
- **Features**: Projections, stats, lineups, next-day results
- **Pricing**: $99/month or $569/year for NBA fantasy data
- **Website**: https://sportsapi.com/api-directory/fantasydata/

### 4. Entity Sports
- **Features**: Comprehensive NBA data, player profiles, real-time fantasy data
- **Pricing**: $200/month or $400/season
- **Website**: https://www.entitysport.com/nba-fantasy-app-api-entity-sports/

### 5. Goalserve
- **Features**: Live scores, player statistics, fantasy data
- **Pricing**: $250/month for NBA/NCAA live data
- **Website**: https://www.goalserve.com

## Free Resources

### RickRunGood NBA Salary Database
- **Free CSV Downloads**: Daily salaries for NBA players
- **Platforms**: DraftKings, FanDuel, Yahoo
- **Format**: CSV files available for download
- **Website**: https://rickrungood.com/nba-salary-database-draftkings/

## Integration Recommendations

### For Personal/Educational Use:
1. **Start with RickRunGood**: Free CSV downloads of daily salaries
2. **SportsDataIO Free Tier**: Access to historical data
3. **Web Scraping**: Some DFS sites may have publicly accessible salary data

### For Production Use:
1. **SportsDataIO**: Best balance of features and pricing
2. **Sportradar**: If you need official NBA salary data
3. **FantasyData.com**: Most comprehensive fantasy-specific data

## What Data You Can Get

### Daily Fantasy Salaries:
- FanDuel salaries
- DraftKings salaries
- Yahoo salaries
- Salary changes day-to-day

### Additional Fantasy Data:
- Ownership projections
- Value ratings (points per dollar)
- Lineup optimizers
- Injury reports
- Starting lineups

## Integration with Current Project

You can integrate salary data to:
1. **Optimize Lineups**: Build optimal lineups within salary cap
2. **Value Analysis**: Calculate points per dollar
3. **Salary-Based Filtering**: Filter players by salary range
4. **Value Picks**: Identify undervalued players

## Next Steps

1. Check RickRunGood for free salary data
2. Sign up for SportsDataIO free tier to test
3. Create a salary integration script
4. Add lineup optimization features


