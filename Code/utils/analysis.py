from huggingface_hub import InferenceClient
import streamlit as st
from typing import List, Dict
import networkx as nx
import pandas as pd
import community as community_louvain
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
import seaborn as sns
import hashlib
import re 


def my_cache(func):
    cache_prefix = f"cache_{func.__name__}"
    
    def wrapper (*args, **kwargs):
        args_str = str(args) + str(kwargs)
        cache_key = f"{cache_prefix}_{hashlib.md5(args_str.encode()).hexdigest()}"

        if cache_key in st.session_state:
            return st.session_state[cache_key]
        
        # Call function and store result
        result = func(*args, **kwargs)
        st.session_state[cache_key] = result
        return result
    
    return wrapper


@my_cache
def analyse_topics(text: str, api_token: str, categories: List[str]) -> Dict:
    try:
        client = InferenceClient(token=api_token,model="facebook/bart-large-mnli")
        
        # Limit to 10
        if len(categories) > 10:
            categories = categories[:10]  
        result = client.zero_shot_classification(
            text,
            candidate_labels=categories,
            multi_label= True
        )
        
        # Debug: show the raw response structure
        st.write("Raw API response:", result)
        
        if isinstance(result, list) and hasattr(result[0], "label"):
            labels = [elem.label for elem in result]
            scores = [elem.score for elem in result]
        else:
            st.error("Unexpected API response format: " + str(result))
            return None
        
        return {
            "text": text,
            "topics": labels[:3], #change both either 0 being the top topic of the post.
            "scores": scores[:3]  
        }
    except Exception as e:
       #  st.error(f"Topic Analysis Error: {str(e)}")
        st.error("Unfortunately the Hugging Face inference API server is having issues errors might occur.")
        return None



def group_analyse_topics(posts: List[str], api_token: str) -> List[Dict]:
    categories = [
        "Technology", "Politics", "Entertainment",
        "Sports", "Personal Life", "Career",
        "Travel", "Food", "Health/Fitness",
        "Education", "Art/Culture", "Science",
    ]
    
    analysed = []
    max_posts = min(len(posts), 10) 
    
    for i, post in enumerate(posts[:max_posts]):   
        analysis = analyse_topics(post, api_token, categories)
        if analysis:
            analysed.append(analysis)
    return analysed

@my_cache
def generate_account_summary(userProfile: dict, posts: list, topic_results: list, api_token: str) -> str:
    # Extract profile information
    profile_desc = userProfile.get("Description", "No description found.")
    name = userProfile.get("Name", "The user")
    followers = userProfile.get("Followers", 0)
    post_count = userProfile.get("Posts", 0)
    
    sample_posts = "\n- \"" + "\"\n- \"".join(post["Text"] for post in posts[:5]) + "\""
    
    top_topics = []
    if topic_results:
        topic_counts = {}
        for result in topic_results:
            for topic in result.get('topics', []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        
        top_topics = [topic for topic, _ in sorted(topic_counts.items(), 
                                                 key=lambda x: x[1], 
                                                 reverse=True)[:3]]
    
    
    avg_likes = sum(post.get("Likes", 0) for post in posts) / len(posts) if posts else 0
    avg_reposts = sum(post.get("Reposts", 0) for post in posts) / len(posts) if posts else 0
    
    prompt = (
        "You are an expert social media analyst. Analyse this Bluesky account and provide a detailed and insightful summary.\n\n"
        "## EXAMPLE ACCOUNT:\n"
        "Name: TechLover\n"
        "Description: Sharing my tech journey and coding adventures\n"
        "Followers: 500\n"
        "Posts: 350\n"
        "Sample Posts:\n"
        "- \"Just got the new MacBook , and the performance is insane!\"\n"
        "- \"Working on a new Python project to help with my coding skills\"\n"
        "- \"Here's my take on the latest AI developments in the industry\"\n"
        "Top Topics: Technology, Personal life\n"
        "Avg Engagement: 25 likes, 5 reposts\n\n"
        "## EXAMPLE SUMMARY:\n"
        "TechLover is a technology-focused Bluesky account with a moderate following of 500 users. They primarily share content about personal tech experiences, coding projects (especially Python), and AI industry trends. Their posts receive good engagement, suggesting they're a respected voice in their tech niche. The account appears to be run by a tech professional or enthusiast who balances personal tech experiences with more educational content.\n\n"
        "## ACTUAL ACCOUNT:\n"
        f"Name: {name}\n"
        f"Description: {profile_desc}\n"
        f"Followers: {followers}\n"
        f"Posts: {post_count}\n"
        f"Sample Posts:\n{sample_posts}\n"
        f"Top Topics: {', '.join(top_topics)}\n"
        f"Avg Engagement: {avg_likes:.1f} likes, {avg_reposts:.1f} reposts\n\n"
        "## ACCOUNT SUMMARY (Write 3-5 sentences that capture the essence of this account, their content focus, audience, and style):"
    )


    
    client = InferenceClient(token=api_token, model="mistralai/Mistral-7B-Instruct-v0.2")
    result = client.text_generation(prompt, max_new_tokens=250, temperature=0.7)

    

    if isinstance(result, dict):
        summary = result.get("generated_text", "")
    else:
        summary = result
    
    # Clean up the summary if needed
    if "## ACCOUNT SUMMARY" in summary:
        summary = summary.split("## ACCOUNT SUMMARY")[1].strip()
    
    return summary

    
def analyse_network(network_data):
    G = nx.DiGraph()
    
    for node in network_data["nodes"]:
        G.add_node(node["id"], **node)
    
    for edge in network_data["edges"]:
        G.add_edge(edge["source"], edge["target"], type=edge["type"])
    
    # Find the central user (the one being analysed)
    central_user = next((node["id"] for node in network_data["nodes"] if node.get("is_central", False)), None)
    
    results = {
        "metrics": {},
        "communities": [],
        "influential_followers": [],
        "interaction_patterns": {},
        "mutual_connections": []
    }
    
    results["metrics"]["node_count"] = len(G.nodes)
    results["metrics"]["edge_count"] = len(G.edges)
    
    if central_user and central_user in G:
        results["metrics"]["in_degree"] = G.in_degree(central_user)  
        results["metrics"]["out_degree"] = G.out_degree(central_user)  
        # Find mutual connections (people who follow and are followed by the central user)
        followers = set(G.predecessors(central_user))
        following = set(G.successors(central_user))
        mutual = followers.intersection(following)
        
        results["mutual_connections"] = [G.nodes[node] for node in mutual if node in G.nodes]
    

    influential = []
    for node_id, node_data in G.nodes(data=True):
        if node_id != central_user and "followers_count" in node_data:
            influential.append({
                "id": node_id,
                "handle": node_data.get("handle", ""),
                "display_name": node_data.get("display_name", ""),
                "followers_count": node_data.get("followers_count", 0)
            })
    
    # Sort by follower count and get top 10
    results["influential_followers"] = sorted(
        influential, 
        key=lambda x: x.get("followers_count", 0), 
        reverse=True
    )[:10]
    

    interactions = network_data["interactions"]
    
    interaction_types = [i["type"] for i in interactions]
    results["interaction_patterns"]["type_counts"] = dict(Counter(interaction_types))

    interactor_counts = Counter([i["source"] for i in interactions])
    top_interactors = interactor_counts.most_common(10)
    
    node_lookup = {node["id"]: node for node in network_data["nodes"]}
    
    results["interaction_patterns"]["top_interactors"] = []
    for did, count in top_interactors:
        if did in node_lookup:
            node_info = node_lookup[did]
            results["interaction_patterns"]["top_interactors"].append({
                "id": did,
                "handle": node_info.get("handle", ""),
                "display_name": node_info.get("display_name", ""),
                "interaction_count": count
            })
        else:
            results["interaction_patterns"]["top_interactors"].append({
                "id": did,
                "handle": "unknown_user",  
                "display_name": "Unknown User", 
                "interaction_count": count
            })
    

    G_undirected = G.to_undirected()
    
    partition = community_louvain.best_partition(G_undirected)
    
    # Group nodes by community
    communitties = {}
    for node, community_id in partition.items():
        if community_id not in communitties:
            communitties[community_id] = []
        if node in G.nodes:
            communitties[community_id].append(G.nodes[node])
    
    results["communities"] = [
        {"id": comm_id, "nodes": nodes}
        for comm_id, nodes in communitties.items()
    ]
    results["communities"].sort(key=lambda x: len(x["nodes"]), reverse=True)
        
    
    return results


def visualise_network_graph(network_data, max_nodes=100):
    
    # Limit number of nodes for performance
    nodes = network_data["nodes"][:max_nodes]
    
    node_ids = [node["id"] for node in nodes]
    central_user_id = next((node["id"] for node in nodes if node.get("is_central", False)), None)
    
    
    edges = [
        edge for edge in network_data["edges"] 
        if edge["source"] in node_ids and edge["target"] in node_ids
    ]
    
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(
            node["id"], 
            name=node.get("display_name", node.get("handle", "Unknown")),
            is_central=node.get("is_central", False)
        )
    
    mutual_connections = set()
    
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        has_reverse = any(e["source"] == target and e["target"] == source for e in edges)
        
        if has_reverse:
            if source > target:  # Only add one direction to avoid duplicates
                mutual_connections.add((target, source))
            else:
                mutual_connections.add((source, target))
    
    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        
        if source == central_user_id:
            relationship = "following"  # Central user follows this account
        elif target == central_user_id:
            relationship = "follower"   # This account follows central user
        else:
            relationship = "other"      # Connection between two non-central users
        
        if (min(source, target), max(source, target)) in mutual_connections:
            relationship = "mutual"
        
        G.add_edge(source, target, relationship=relationship)
    pos = nx.spring_layout(G, seed=42)
    
    mutual_x, mutual_y = [], []
    following_x, following_y = [], []
    follower_x, follower_y = [], []
    other_x, other_y = [], []

    mutual_texts, following_texts, follower_texts, other_texts = [], [], [], []
    
    for edge in G.edges(data=True):
        source, target, data = edge
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        relationship = data.get('relationship', 'other')
        
        # Create hover text for the edge
        source_name = G.nodes[source].get('name', source)
        target_name = G.nodes[target].get('name', target)
        
        if relationship == "mutual":
            hover_text = f"{source_name} and {target_name} follow each other"
            mutual_x.extend([x0, x1, None])
            mutual_y.extend([y0, y1, None])
            mutual_texts.append(hover_text)
        elif relationship == "following":
            hover_text = f"Central user follows {target_name}"
            following_x.extend([x0, x1, None])
            following_y.extend([y0, y1, None])
            following_texts.append(hover_text)
        elif relationship == "follower":
            hover_text = f"{source_name} follows central user"
            follower_x.extend([x0, x1, None])
            follower_y.extend([y0, y1, None])
            follower_texts.append(hover_text)
        else:
            hover_text = f"{source_name} → {target_name}"
            other_x.extend([x0, x1, None])
            other_y.extend([y0, y1, None])
            other_texts.append(hover_text)
    
    mutual_trace = go.Scatter(
        x=mutual_x, y=mutual_y, 
        line=dict(width=1.5, color='#28a745'), 
        hoverinfo='text', 
        mode='lines', 
        name='Mutual Follows',
        hovertext=mutual_texts
    )
    
    following_trace = go.Scatter(
        x=following_x, y=following_y, 
        line=dict(width=1.0, color='#007bff'), 
        hoverinfo='text', 
        mode='lines', 
        name='Following',
        hovertext=following_texts
    )
    
    follower_trace = go.Scatter(
        x=follower_x, y=follower_y, 
        line=dict(width=1.0, color='#dc3545'), 
        hoverinfo='text', 
        mode='lines', 
        name='Follower',
        hovertext=follower_texts
    )
    
    other_trace = go.Scatter(
        x=other_x, y=other_y, 
        line=dict(width=0.5, color='#6c757d'), 
        hoverinfo='text', 
        mode='lines', 
        name='Other',
        hovertext=other_texts
    )
    
    node_x = []
    node_y = []
    node_colours = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Set node color based on whether it's the central user
        is_central = G.nodes[node].get('is_central', False)
        
        has_mutual = False
        # Check both possible orderings of the tuple
        if (node, central_user_id) in mutual_connections or (central_user_id, node) in mutual_connections:
            has_mutual = True
        
        if is_central:
            node_colours.append('#1DA1F2')  # Blue for central user
        elif has_mutual:
            node_colours.append('#28a745')  # Green for mutual connections
        else:
            node_colours.append('#aaaaaa')  # Gray for others
        
        node_sizes.append(20 if is_central else 12 if has_mutual else 8)
        
        node_text.append(G.nodes[node].get('name', node))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_colours,
            size=node_sizes,
            line=dict(width=1, color='#888')
        ),
        name='Accounts'
    )
    
    # Collect all non-empty traces
    data = []
    if mutual_x:  
        data.append(mutual_trace)
    if following_x:
        data.append(following_trace)
    if follower_x:
        data.append(follower_trace)
    if other_x:
        data.append(other_trace)
    data.append(node_trace)  

    # Create the figure
    fig = go.Figure(
        data=data,
        layout=go.Layout(
            title={'text': 'Network Graph', 'font': {'size': 16}},
            showlegend=True,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.8)"
            )
        )
    )
    
    return fig
def create_network_dashboard(network_data, analysis_results):
    st.subheader("Network Overview")
    with st.expander("How to read this network visualisation"):
        st.markdown("""
        Nodes represent accounts in the network:
        - The blue node is the account being analysed (central account)
        - The green nodes are accounts that have mutual following with the central account
        - The grey nodes are other acconts that are connected to the central account
                    
        The lines represent the connection type between the nodes:
        - The green lines are mutual connections
        - The blue lines show what accounts the central account follows
        - The red lines show what accounts follow the central account
        - The grey lines are connections between other accounts within the network
        
        What can you take from this:
        Nodes positioned close together in clusters can represent communities with similar interests
        Nodes placed in the centre will usually have more connections than those at the edges
        Nodes that are closer together are more connected
        
        Knowing this sepearate clusters in the network may suggest different communities/social circles
        Many red lines may indiciate a high level of influence from the central account
        A variety of connection types can show that the central account has a active involvement in their communities
        """)
        
        st.info("Hover over any node to see the account name, and over any line to see the relationship type.")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Network Size", analysis_results["metrics"].get("node_count", 0))
    
    with col2:
        st.metric("Followers", analysis_results["metrics"].get("in_degree", 0))
    
    with col3:
        st.metric("Following", analysis_results["metrics"].get("out_degree", 0))
    
    st.subheader("Network Visualisation")
    network_fig = visualise_network_graph(network_data)
    
    st.plotly_chart(network_fig, use_container_width=True)
    

    st.subheader("Interaction Patterns")
    
    counts = analysis_results["interaction_patterns"]["type_counts"]
    
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values(), color=['#1DA1F2', '#17BF63', '#F45D22'])
    ax.set_title("Interaction Types")
    ax.set_xlabel("Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    
    st.write("### Most Frequent Interactors")
    st.info("Please note if you do see unknown users it is because the current network size does not contain the information of the user.")
    df_interactors = pd.DataFrame(analysis_results["interaction_patterns"]["top_interactors"])
    
    df_interactors = df_interactors[["display_name", "handle", "interaction_count"]]
    df_interactors.columns = ["Name", "Handle", "Interactions"]
            
    st.dataframe(df_interactors)
    
    # Display mutual connections
    st.subheader("Mutual Connections")
     # will check if there are even any mutual connections within the analysed network.

    if analysis_results["mutual_connections"]:
        
        df_mutual = pd.DataFrame(analysis_results["mutual_connections"])
        
        df_mutual = df_mutual[["display_name", "handle"]]
        df_mutual.columns = ["Name", "Handle"]
        st.dataframe(df_mutual)
    else:
        st.info("No mutual connections were found within the network size scope.")



def visualize_shared_interests(shared_interests_data):
    st.subheader("Shared Interest Heatmap")
    with st.expander("How to read this heatmap"):
        st.markdown("""
        - Each row is an account with the user being the main account.
        - Each column is a topic that is identified from the posts.
        - The number of posts that is of the genre of the topic.
        - A Darker blue suggests a high amount of posts in that topic.
                
        What can you take from this:
        - Similar patterns: Followeers that are similar to the main account will have similar colours for the topics.
        - Dark cells in the same columns: Topics that are popular amongst followers.
        - Empty/light cells: These are topics that are not popular with the account.
        - Contrast: Differences in topic preferences between the main account and followers.
        
            
        
        """)

    Total_topics = set()
    for topic in shared_interests_data["user_topics"].keys():
        Total_topics.add(topic)
    
    for follower in shared_interests_data["follower_analyses"]:
        for topic in follower["topics"].keys():
            Total_topics.add(topic)
    
    topics_list = list(Total_topics)
    followers = shared_interests_data["follower_analyses"]
    

    heatmap_data = []

    user = {"name": "User"}
    for topic in topics_list:
        user[topic] = shared_interests_data["user_topics"].get(topic, 0)
    heatmap_data.append(user)
    
    for follower in followers:
        follower_row = {"name": follower["handle"]}
        for topic in topics_list:
            follower_row[topic] = follower["topics"].get(topic, 0)
        heatmap_data.append(follower_row)
    
    df = pd.DataFrame(heatmap_data)
    df = df.set_index("name")

    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
    ax.set_title("Topic Frequency: User and Followers")
    st.pyplot(fig)
    
@my_cache
def analyse_content_uniquenesses(posts: list, api_token: str) -> dict:
    content_posts = [p['Text'] for p in posts[:10]]
    content_text = "\n- \"" + "\"\n- \"".join(content_posts) + "\""
    prompt = (
        "As a social media content analyst, evaluate the uniqueness and distinctiveness of the following "
        "social media posts from a single user. Consider:\n"
        "1. How original is their perspective compared to common viewpoints?\n"
        "2. Do they use distinctive language, terminology, or phrasing?\n"
        "3. Are they discussing niche topics or mainstream subjects?\n"
        "4. Is their writing style unique or does it follow common patterns?\n\n"
        f"POSTS TO ANALYZE:\n{content_text}\n\n"
        "First, rate the overall content uniqueness on a scale of 1-10, where 1 is extremely common "
        "content and 10 is highly distinctive and original.\n\n"
        "Then provide a brief analysis (3-4 sentences) of what makes this user's content unique or common.\n\n"
        "Output format:\n"
        "Uniqueness Score: [1-10]\n"
        "Analysis: [your analysis here]"
    )
    client = InferenceClient(token=api_token, model="mistralai/Mistral-7B-Instruct-v0.2")
    result = client.text_generation(prompt, max_new_tokens=300, temperature=0.7)

    # Parse the result to extract score and analysis
    results_text = result
    score_result = re.search(r"Uniqueness Score:\s*(\d+(?:\.\d+)?)", results_text)
    score = float(score_result.group(1))
    
    analysis_result = re.search(r"Analysis:(.*?)(?:\n\n|$)", results_text, re.DOTALL)
    analysis = analysis_result.group(1).strip() 
    
    return {
        "score": score,
        "analysis": analysis,
        "full_response": results_text
    }
