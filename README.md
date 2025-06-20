# Social Media Analysis Tool 
A social media analytics dashboard that provides deep insights into user profiles, content patterns, and social networks making use of AI models from the HuggingFace Inference API for certain parts of the analysis. Built with Python and Streamlit.

## Features:

### Profile Analysis:

- Profile Summaries: Create detailed summaries about user accounts using HuggingFace AI models.
- Content Uniqueness Scoring: Determine the uniqueness and distinctiveness of user content From 1-10, 1 being not unique and distinctive and 10 being highly unique and distinctive.
- Topic Classification: Uses zero-shot classification to help categorise posts into predefined topics. 
- Visual Topic Distribution: Bar Chart to showcase content themes.

### Network Analysis:

-Interactive Network Visualization: Look at social connections with colour coded relationship types.
-Community Detection: Identify clusters and communities using the Louvain algorithm.
-Mutual Connection Analysis: Discover bidirectional relationships within the network.
-Influence Metrics: Analyze follower counts and interaction patterns.
-Relationship Mapping: Visualize following/follower dynamics.

### Follower Analysis:

-Shared Interest Detection: Using the same topic classification but on followers to find shared interests between the analysed user and its followers.
-Interactive Heatmaps: Shared interests will be represented as a heatmap.

## Technology Stack

-Frontend: Streamlit
-API Integration: atproto (Bluesky), Hugging Face Inference API
-Data Processing: pandas, NetworkX
-AI/ML: Pre-trained models via Hugging Face (BART, Mistral-7B)
-Visualization: Plotly, Matplotlib, Seaborn
-Network Analysis: NetworkX, python-louvain


