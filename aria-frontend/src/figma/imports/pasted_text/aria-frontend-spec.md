Design and build the complete frontend UI/UX for **ARIA**, an AI-powered Business Analytics and Intelligence platform.

ARIA is not a landing page or a simple chatbot. It is a complete professional analytics application used to ask business questions in natural language, analyze structured business data, discover insights, perform predictive analysis, generate prescriptions/recommendations, test hypotheses, and interact with an AI copilot.

The final product should feel like a **real production-grade SaaS analytics platform**, suitable for a university FYP demo and professional portfolio.

## 1. OVERALL DESIGN DIRECTION

Create a premium, modern, intelligent enterprise analytics interface.

Visual inspiration:

* Linear
* Vercel
* Perplexity
* modern enterprise analytics platforms
* sophisticated AI products

Do NOT make it look like:

* a generic ChatGPT clone
* a generic admin dashboard
* a template with dozens of colorful cards
* an overly futuristic neon AI interface
* a cryptocurrency dashboard

ARIA should feel:

* intelligent
* trustworthy
* analytical
* elegant
* minimal
* highly polished
* technically sophisticated
* easy to understand

Prioritize excellent hierarchy, whitespace, typography, subtle depth, clear data visualization and meaningful interaction.

Use a sophisticated neutral foundation with a restrained accent color system.

Support both:

* Light mode
* Dark mode

Use subtle transitions and micro-interactions rather than excessive animations.

---

# 2. GLOBAL APPLICATION STRUCTURE

Create a persistent compact sidebar navigation.

Sidebar items:

1. Dashboard
2. AI Analytics
3. Insights
4. Predictive Analysis
5. Prescription
6. Hypothesis
7. Data
8. Copilot
9. History
10. Settings

The sidebar should be compact and elegant.

Include:

* ARIA logo
* navigation icons
* active-state indicator
* tooltips when collapsed
* collapse/expand interaction
* user profile at bottom
* settings access

The main content area should adapt when the sidebar is collapsed.

Create a consistent top navigation system with:

* current page title
* contextual actions
* notifications
* theme toggle
* profile menu

---

# 3. ARIA DESIGN SYSTEM

Create reusable components and use them consistently across every page.

Components should include:

* Primary button
* Secondary button
* Ghost button
* Icon button
* Search input
* Natural-language query input
* Dropdown
* Select
* Tabs
* Segmented controls
* Cards
* Metric cards
* Chart containers
* Tables
* Badges
* Status indicators
* Tooltips
* Modals
* Drawers
* Toast notifications
* Breadcrumbs
* Pagination
* Progress indicators
* Skeleton loaders
* Empty states
* Error states
* Confirmation dialogs
* AI response blocks
* Insight cards
* Recommendation cards
* Prediction cards
* Evidence blocks
* SQL/code blocks
* Data-source indicators

Create a coherent spacing system, typography system, radius system and elevation system.

Do not randomly vary component styles between pages.

---

# 4. DASHBOARD

Create a compact but highly intelligent ARIA dashboard.

The dashboard should immediately communicate:
"What is happening in my business?"

Top section:

Greeting:
"Good morning"

Supporting text:
"Here's what ARIA found in your business data."

Primary feature:
A large natural-language AI query box.

Placeholder:
"Ask ARIA anything about your business..."

Examples underneath:

* "Why did sales decrease last month?"
* "Which products are performing best?"
* "Forecast next month's revenue"
* "Find unusual changes in sales"

Include autocomplete suggestions similar to a modern search engine.

As the user types, display intelligent query suggestions.

Below the query interface:

## Business Overview

Create elegant KPI cards such as:

Revenue
Sales
Orders
Customers
Profit

Each card should include:

* current value
* percentage change
* comparison period
* miniature trend visualization
* positive/negative indicator

Do not overcrowd the dashboard.

## AI Highlights

Create an "ARIA Intelligence" section.

Show automatically generated findings such as:

"Revenue increased 14.8% this month."

"Electronics generated the strongest growth."

"Sales in Region X are significantly below the expected trend."

Each insight should have:

* concise explanation
* severity/importance
* supporting metric
* "Explore" action

## Analytics Overview

Create sophisticated charts for:

* Revenue trend
* Sales trend
* Category performance
* Regional performance

Charts should be clean and professional.

## Anomalies

Display detected anomalies with:

* metric
* date
* deviation
* explanation
* severity

## Recommended Questions

ARIA should proactively suggest useful questions.

Example:
"Want to understand why revenue changed?"

Include quick action buttons.

---

# 5. AI ANALYTICS PAGE

This is ARIA's main natural-language analytics workspace.

Do NOT redirect the user to the dashboard when performing analysis.

The analysis should happen directly inside this page.

Top:

Page title:
"AI Analytics"

Subtitle:
"Ask questions. Explore your data. Get answers."

Large query composer.

Features:

* autocomplete
* query suggestions
* voice/search-style interaction placeholder
* submit button
* clear button
* recent queries

Example query:

"Which product category generated the highest revenue in Q3?"

When submitted, show an elegant processing state.

Processing stages:

Understanding question
↓
Inspecting data
↓
Generating analysis
↓
Preparing result

Then display the result.

## Result layout

Show:

### Answer

A clear natural-language answer generated by ARIA.

### Visualization

Automatically select an appropriate chart.

### Supporting data

Display a table containing the relevant records/aggregations.

### Generated SQL

Provide a collapsible SQL section.

Use a beautiful code block.

Include:

* Copy
* Expand
* Explain SQL

### AI Explanation

Explain the result in simple business language.

### Follow-up questions

Generate contextual follow-ups such as:

"Why did Electronics perform better?"

"Show me the trend over the last 12 months."

"Compare this with the previous quarter."

The user should be able to continue analysis naturally.

---

# 6. INSIGHTS PAGE

Create a dedicated intelligence discovery page.

This page should NOT simply repeat the dashboard.

ARIA automatically discovers meaningful patterns in the business data.

Top controls:

* date range
* data source
* category
* severity
* insight type

Insight categories:

* Trends
* Anomalies
* Correlations
* Growth
* Decline
* Outliers
* Opportunities
* Risks

Create a sophisticated insight feed.

Each insight card should include:

Title
Short explanation
Affected metric
Magnitude
Confidence
Supporting visualization
Evidence
Date detected

Example:

"Electronics revenue is growing significantly faster than other categories."

Show the relevant chart beside it.

Allow:

* Explore
* Ask ARIA
* Save
* Share

Create a detailed insight view with:

Observation
Evidence
Possible causes
Business impact
Related metrics
Recommended next analysis

---

# 7. PREDICTIVE ANALYSIS PAGE

Create a forecasting workspace.

Top:

"Predictive Analysis"

Subtitle:
"Understand what is likely to happen next."

Configuration area:

Metric
Dataset
Time range
Forecast horizon
Granularity

Allow users to configure the prediction.

Main visualization:

Historical data + forecast.

Clearly distinguish:

* historical values
* predicted values
* confidence interval

Show prediction summary:

Expected value
Expected growth
Upper bound
Lower bound
Confidence

Create:

## Prediction Drivers

Display factors contributing to the prediction.

## Scenario Analysis

Allow:

Base case
Optimistic
Pessimistic

Display the scenarios visually.

## AI Interpretation

Explain what the forecast means for the business.

Include caveats and uncertainty.

---

# 8. PRESCRIPTION PAGE

This page converts analysis into recommended actions.

Title:

"Prescription"

Subtitle:

"Turn insights into actionable decisions."

Create recommendation cards.

Each recommendation should contain:

Problem
Evidence
Recommended action
Expected impact
Priority
Confidence
Potential risk

Example:

"Reduce inventory allocation for low-performing products."

Include:

Expected revenue impact
Expected cost impact
Implementation difficulty
Risk level

Create a clear visual hierarchy.

Include alternative recommendations.

Allow the user to:

Accept
Save
Explore evidence
Ask ARIA

---

# 9. HYPOTHESIS PAGE

Create a workspace for testing business hypotheses.

Title:

"Hypothesis"

Subtitle:

"Test assumptions against your data."

Create a natural-language hypothesis input.

Example:

"Does increasing discounts increase sales?"

Then show configuration:

Dataset
Variables
Time period
Filters

When running:

Show an elegant analysis progress state.

Results page should contain:

Hypothesis
Method
Variables
Sample/data used
Result
Statistical evidence
Confidence
Interpretation

Clearly communicate:

Supported
Not supported
Inconclusive

Avoid overly technical presentation.

Allow "Explain this result" through ARIA.

---

# 10. DATA PAGE

Create a professional data management/database interface.

Title:

"Data"

Sections:

Connected Sources
Schema
Tables
Data Preview
Data Quality

Show connected database cards.

Each connection should display:

Database type
Connection status
Tables
Last synchronized
Health

Create a schema explorer.

Display:

Tables
Columns
Relationships
Data types

Create relationship visualization.

Allow users to inspect table structures.

Data preview should include:

* column names
* data types
* sample rows
* null values
* quality indicators

Create a Data Quality section with:

Missing values
Duplicates
Invalid values
Outliers
Completeness

---

# 11. COPILOT PAGE

Create ARIA's conversational AI workspace.

This should feel more intelligent than a normal chatbot.

Left side:
Conversation list.

Main area:
Conversation.

Messages should support:

* text
* tables
* charts
* insights
* SQL
* recommendations
* analysis results

ARIA should understand context across multiple questions.

Example:

User:
"Show me revenue for electronics."

ARIA:
[chart + explanation]

User:
"Why did it decrease?"

ARIA should understand "it" refers to electronics revenue.

Add contextual suggestion chips.

Create an elegant AI response design.

---

# 12. HISTORY PAGE

History must be its own separate section.

Do NOT place the complete history system on the dashboard.

Create a searchable history workspace.

Categories:

All
Analytics
Insights
Predictions
Prescriptions
Hypotheses

Each history item should show:

Question/title
Type
Date
Data source
Short result summary

Include:

Search
Filters
Date filtering
Sort
Delete
Save
Open

Create empty state and loading state.

---

# 13. SETTINGS PAGE

Create a professional settings interface.

Sections:

Profile
Appearance
AI Configuration
Data Sources
Notifications
Privacy
Application

## AI Configuration

Allow the user to select:

Cloud LLM
Local LLM

Create a clear model selection interface.

Show:

Model name
Provider
Status
Performance
Availability

Do not make the configuration intimidating.

## Appearance

Light
Dark
System

## Notifications

Analysis completed
New insights
Prediction updates
System alerts

---

# 14. AI STATES

Design detailed states for ARIA.

Loading:

ARIA is thinking...

Then show intelligent progress steps.

Empty:

"No analysis yet."

Error:

"We couldn't complete this analysis."

Include:

* explanation
* retry
* modify query
* report problem

No results:

"We couldn't find enough relevant data."

Include suggestions.

---

# 15. RESPONSIVE DESIGN

Create desktop-first layouts.

Also design responsive behavior for:

Desktop
Tablet
Mobile

The sidebar should transform appropriately.

Charts must remain readable.

Tables should become horizontally scrollable or transform into cards.

The query interface must remain prominent on smaller screens.

---

# 16. MICRO-INTERACTIONS

Use subtle professional animations.

Examples:

* sidebar transitions
* card hover
* chart reveal
* query submission
* AI processing
* insight expansion
* modal transitions
* page transitions
* button feedback
* skeleton loading

Animations should feel smooth and restrained.

Do NOT use excessive glowing effects.

---

# 17. ACCESSIBILITY

Ensure:

* strong contrast
* readable typography
* clear focus states
* keyboard-friendly controls
* understandable icons
* tooltips
* meaningful labels

---

# 18. IMPORTANT PRODUCT PRINCIPLES

ARIA must feel like an intelligent analytics assistant rather than a collection of disconnected pages.

The user should always understand:

1. What ARIA found
2. Why ARIA found it
3. What data supports it
4. How confident ARIA is
5. What the user can do next

Every major AI-generated result should therefore have:

Answer
Evidence
Visualization
Explanation
Next action

---

# 19. FINAL QUALITY BAR

Before considering the design complete, review the entire application as a senior product designer.

Fix:

* inconsistent spacing
* inconsistent typography
* weak hierarchy
* excessive cards
* unnecessary decoration
* repetitive layouts
* poor chart choices
* unclear navigation
* confusing AI states
* missing empty/error/loading states
* inconsistent buttons
* inconsistent component styling

The final result should look like a **real commercial AI analytics product**, not an FYP template.

Make the interface visually impressive enough for a final-year project presentation while keeping it genuinely usable.

Most importantly:

**Design the entire ARIA application as ONE unified product and design system.**

Do not make each page look like it came from a different template.

Create all pages, reusable components, states, interactions, responsive layouts and the complete navigation flow.
