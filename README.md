# CareerLens 
A workforce intelligence platform we built. The idea is simple — the U.S. job market has decades of public data sitting around (BLS, O*NET) that most people never actually look at. We wanted to make it actually useful and visual, not just a bunch of CSV files.

---

## What it does

You pick a profession and a state, and the app shows you everything about that career — salary trends from 2005 to 2024, how it compares across all 50 states, how risky it is to automation, and where it's headed by 2030. Then we layered AI on top so you can just ask it questions in plain English.

There are 6 main sections:

**State Map** — All 50 states on an interactive map. Hover over any state and it shows you the median salary, employment growth, and which "cluster" that state falls into (High Compensation Hub, Emerging Growth Market, etc.). States are color coded so you can immediately see patterns.

**20-Year Heatmap** — A grid showing 2005 through 2024 for whatever metric you care about. Salary, demand, stress, automation risk, stability. Hover a cell and it gives you a specific insight for that year — like why 2020 was such a weird year for nurses, or why 2015 mattered for tech.

**Tradeoff Radar** — Pick 3 professions and compare them on a spider chart across 6 dimensions: salary, demand, stability, remote feasibility, stress, and automation risk. Good for when you're deciding between careers and want to see the actual tradeoffs visually.

**Trajectory Matcher** — This is the Actian VectorAI part. You pick a profession and it searches through historical snapshots to find the closest matches. So it might tell you "Data Scientist in 2024 looks a lot like Software Developer in 2014" based on the growth curve, salary trajectory, and demand patterns. Cosine similarity across 20 years of data.

**Forecast 2030** — Uses the historical trend data to project salaries and demand out to 2030. The chart clearly separates real BLS data from AI-estimated projections so you know what's fact vs. what's a model output.

**Ask AI** — Powered by Gemini. You can type any question like "why did nurse salaries spike in 2020" or "compare tech jobs in California vs Texas" and it gives you a structured analytical response. Not a chatbot vibe — more like a consulting report.

---

## Why we built it this way

Most career info online is either super outdated or just vibes ("software engineering is a great field!"). We wanted actual numbers, actual trends, and actual geographic differences. A nurse in Mississippi makes almost 40% less than a nurse in California — that matters and most people don't know it. We also wanted it to be something judges could actually play with, not just watch a demo of. So every dropdown, every hover, every button does something real.

---

## Tech stack

- **Frontend** — HTML, CSS, JavaScript. No framework, kept it simple so it runs anywhere without setup.
- **Charts** — Chart.js for the radar and forecast charts, D3.js for the state map
- **AI** — Gemini API for natural language analysis and trend interpretation
- **Vector DB** — Actian VectorAI for the trajectory similarity search across historical career snapshots
- **Data** — Based on BLS Occupational Employment Statistics and O*NET Work Context metrics (2005–2024)
