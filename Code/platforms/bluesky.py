import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from atproto import Client
import re
from collections import defaultdict
from platforms.platform_base import Platform_Analyser
from utils.analysis import *

class Bluesky_Analyser(Platform_Analyser):
    def __init__(self):
        super().__init__()
        self.client = Client()
        self.authenticate()
        self.hf_token = st.secrets["HF_API_TOKEN"]

    def authenticate(self):
        try:
            self.client.login(  
                st.secrets["BLUESKY_USERNAME"], st.secrets["BLUESKY_PASSWORD"]
            )
        except Exception as e:
            st.error(f"Authentication has failed: {str(e)}")

    def handle_validation(self, handle: str) -> str:
        
        if handle.startswith("@"):
            handle = handle[1:]
        if '.' not in handle:
            return f"{handle}.bsky.social"  # default domain
        
        pattern = r"^[a-zA-Z0-9_.-]+\.[a-zA-Z0-9.-]+$"
        
        if re.match(pattern, handle):
            return handle
        else:
            st.error("Invalid username format try again.")
            return None


    def text_cleaner(self, text: str) -> str:
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'#\w+', '', text)
        return text.strip()

    def get_bluesky_data(self, handle: str):
        handle = self.handle_validation(handle)

        if not hasattr(self.client, 'me'):
            st.error("Not authenticated with Bluesky")
            return None

        userProfile = self.client.get_profile(handle)
        feed = self.client.get_author_feed(
            userProfile.did,
            filter="posts_no_replies"
            )
        posts = [{
            "Text": self.text_cleaner(post.post.record.text),   
            "Likes": post.post.like_count,
            "Reposts": post.post.repost_count,
            "Date": pd.to_datetime(post.post.record.created_at).strftime('%Y-%m-%d') #obtain date created which is a timestamp string and change it to datetime format
        } for post in feed.feed]

        followers = self.client.get_followers(userProfile.did).followers
        
        return {
            "profile": {
                "Name": userProfile.display_name,
                "Handle": userProfile.handle,
                "Followers": userProfile.followers_count,
                "Posts": userProfile.posts_count,
                "Description": userProfile.description
            },
            "posts": posts,
            "followers": followers
        }
    def get_network_data(self, handle: str, max_depth: int = 1, max_connections: int = 10):
        try:
            handle = self.handle_validation(handle)

        
            userProfile = self.client.get_profile(handle)
            userProfile_did = userProfile.did
            
            # network data structure
            network = {
                "nodes": [], 
                "edges": [],  
                "interactions": []  
            }
            
            # Central node will be the analysed account
            network["nodes"].append({
                "id": userProfile_did,
                "handle": userProfile.handle,
                "display_name": userProfile.display_name,
                "followers_count": userProfile.followers_count,
                "follows_count": userProfile.follows_count,
                "posts_count": userProfile.posts_count,
                "is_central": True  # Mark as the central user
            })
            
            # Get followers whilist following the max connections
            followers = self.client.get_followers(userProfile_did, limit=max_connections).followers
            for follower in followers:
                network["nodes"].append({
                    "id": follower.did,
                    "handle": follower.handle,
                    "display_name": follower.display_name,
                    "is_central": False
                })
                network["edges"].append({
                    "source": follower.did,
                    "target": userProfile_did,
                    "type": "follows"
                })
            
            
            following = self.client.get_follows(userProfile_did, limit=max_connections).follows
            for follow in following:
                # Check if node is already in the network (if its a follower)
                if not any(node["id"] == follow.did for node in network["nodes"]):
                    network["nodes"].append({
                        "id": follow.did,
                        "handle": follow.handle,
                        "display_name": follow.display_name,
                        "is_central": False
                    })
                network["edges"].append({
                    "source": userProfile_did,
                    "target": follow.did,
                    "type": "follows"
                })
            
            
            # analyse interactions from recent posts
            feed = self.client.get_author_feed(userProfile_did, limit=50).feed
            
            for post_item in feed:
                post = post_item.post
                post_uri = post.uri
                try:
                    likes = self.client.get_likes(post_uri).likes
                    for like in likes:
                        network["interactions"].append({
                            "source": like.actor.did,
                            "target": userProfile_did,
                            "post_uri": post_uri,
                            "type": "like",
                            "timestamp": post.indexed_at
                        })
                except Exception as e:
                    print(f"Error fetching likes: {e}")
                try:
                    reposts = self.client.get_reposted_by(post_uri).reposted_by
                    for repost in reposts:
                        network["interactions"].append({
                            "source": repost.did,
                            "target": userProfile_did,
                            "post_uri": post_uri,
                            "type": "repost",
                            "timestamp": post.indexed_at
                        })
                except Exception as e:
                    print(f"Error fetching reposts: {e}")
                if hasattr(post_item, 'replies') and post_item.replies:
                    for reply in post_item.replies:
                        network["interactions"].append({
                            "source": reply.post.author.did,
                            "target": userProfile_did,
                            "post_uri": reply.post.uri,
                            "type": "reply",
                            "timestamp": reply.post.indexed_at
                        })
                interactor_dids = set()
                for interaction in network["interactions"]:
                    interactor_dids.add(interaction["source"])

                existing_dids = {node["id"] for node in network["nodes"]}
                missing_dids = interactor_dids - existing_dids

                #Attepmts to obtain handles outside of analysis scope most of the time it wont work due to the unknown handles being outside the network size
                missing_dids_list = list(missing_dids)[:50]  
                for did in missing_dids_list:
                    try:
                        userProfile = self.client.get_profile_by_did(did)
                        if userProfile:
                            network["nodes"].append({
                                "id": did,
                                "handle": userProfile.handle if hasattr(userProfile, "handle") else "unknown",
                                "display_name": userProfile.display_name if hasattr(userProfile, "display_name") else "",
                                "is_central": False
                            })
                    except Exception as e:
                       # print(f"Could not fetch profile for {did}: {e}")
                       continue
            
            return network
        
            
        except Exception as e:
            st.error(f"Error fetching network data: {str(e)}")
            return None

    def run(self):
        st.title("Bluesky Profile Analyser")
        
        tabs = st.tabs(["Profile Analysis", "Network Analysis","Follower Analysis"])
        
        handle = st.sidebar.text_input("Enter a Bluesky handle (e.g., username.bsky.social):")
        
        if handle:
            with st.spinner("Fetching data..."):
                user_profile_data = self.get_bluesky_data(handle)
            
            if not user_profile_data:
                st.error("Could not get profile data. Please check handle and try again.")
                return
            
            with tabs[0]:
                self.run_profile_overview(user_profile_data)
            with tabs[1]:
                self.run_network_analysis(handle)
            with tabs[2]:
                self.run_follower_analysis(handle)
        else:
            st.info("Enter a Bluesky handle to begin analysis")

    def run_profile_overview(self, data):
        st.subheader("Profile Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Followers", data['profile']['Followers'])
        with col2:
            st.metric("Total Posts", data['profile']['Posts'])
        with col3:
            st.write("Bio:", data['profile']['Description'])
        
        
        
        
        with st.spinner("Analysing... (Can take 30+ seconds)"):
            posts_to_analyse = [post["Text"] for post in data['posts']]
            topic_results = group_analyse_topics(posts_to_analyse, self.hf_token)
        
        
        st.subheader("Profile Summary")
        summary = generate_account_summary(
            data["profile"], 
            data["posts"], 
            topic_results,
            self.hf_token
        )
        
        if summary:
            st.write(summary)
        else:
            st.warning("Cant genearate summary right now.")

        st.subheader("Content Uniqueness")
        uniqueness_result = analyse_content_uniquenesses(data['posts'], self.hf_token)
    
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Uniqueness Score", f"{uniqueness_result['score']:.1f}/10")
        with col2:
            st.write(uniqueness_result["analysis"])
        
        topic_results = []
        with st.spinner("Analysing... (Can take 30+ seconds)"):
            posts_to_analyse = [post["Text"] for post in data['posts']]
            topic_results = group_analyse_topics(posts_to_analyse, self.hf_token)
        
        if topic_results:
            # Calculate topic freq
            topic_counts = defaultdict(int)
            for result in topic_results:
                for topic in result.get('topics', []):
                    topic_counts[topic] += 1
            handle = data['profile']['Handle']
            st.session_state[f"user_topic_counts_{handle}"] = topic_counts
            
            st.write("Most Frequent Topics")
            
            # Sort topics by freq
            sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
            
            if sorted_topics:
                # Get top 5 topics or all if less than 5
                top_topics = sorted_topics[:min(5, len(sorted_topics))]
            
                topic_names = [t[0] for t in top_topics]
                topic_values = [t[1] for t in top_topics]
                
                fig, ax = plt.subplots()
                bars = ax.barh(
                    y=topic_names,  # Use the topic names for y-axis
                    width=topic_values,  # Use the counts for bar width
                    color='#198bd1'
                )
                ax.set_xlabel("Number of Posts")
                ax.set_title("Top Content Themes")
                st.pyplot(fig)
            else:
                st.info("Too little topics to visualise")
            
        else:
            st.warning("No topics could be identified in these posts")           
            

    def run_network_analysis(self, handle):
        tabs = st.tabs(["Network Structure"])
        with tabs[0]:
            with st.container():
                max_connections = st.slider("Maximum connections to analyse", 10, 100, 50)
            
            with st.spinner("Analysing social network... (may take some time)"):
                network_data = self.get_network_data(handle, max_connections=max_connections)
                
                if network_data:
                    analysis_results = analyse_network(network_data)
                    
                    if analysis_results:
                        create_network_dashboard(network_data, analysis_results)
                    else:
                        st.error("Cant analyse network data")
                else:
                    st.error("Couldnt get network data")
    def get_follower_post(self, follower_did, max_posts=5):
        follower_posts = self.client.get_author_feed(follower_did, limit=max_posts)
        posts = [{
            "Text": self.text_cleaner(post.post.record.text),   
            "Likes": post.post.like_count,
            "Reposts": post.post.repost_count,
            "Date": pd.to_datetime(post.post.record.created_at).strftime('%Y-%m-%d')
        } for post in follower_posts.feed]
        return posts

    def analyse_shared_interests(self, handle):
        user_profile_data = self.get_bluesky_data(handle)
        
        user_posts = [post["Text"] for post in user_profile_data["posts"]]
        user_topics = group_analyse_topics(user_posts, self.hf_token)
        
        user_topic_counts = defaultdict(int)
        for result in user_topics:
            for topic in result.get('topics', []):
                user_topic_counts[topic] += 1
        
        followers = user_profile_data["followers"][:3]  # test diff numbersto see how high to put it
        
        follower_analyses = []
        

        for follower in followers:
            follower_posts = self.get_follower_post(follower.did)
                
            follower_post_texts = [post["Text"] for post in follower_posts]
            follower_topics = group_analyse_topics(follower_post_texts, self.hf_token)
            
            follower_topic_counts = defaultdict(int)
            for result in follower_topics:
                for topic in result.get('topics', []):
                    follower_topic_counts[topic] += 1
            
            shared_topics = []
            for topic, count in follower_topic_counts.items():
                if topic in user_topic_counts:
                    shared_topics.append({
                        "topic": topic,
                        "user_count": user_topic_counts[topic],
                        "follower_count": count
                    })
            
            follower_analyses.append({
                "did": follower.did,
                "handle": follower.handle,
                "display_name": follower.display_name,
                "topics": dict(follower_topic_counts),
                "shared_topics": shared_topics
            })
        
        return {
            "user_topics": dict(user_topic_counts),
            "follower_analyses": follower_analyses
        }
    def run_follower_analysis(self, handle):
        st.subheader("Follower Interest Analysis")
        st.info("This analysis looks at post topics to find shared interest.")
        
        with st.spinner("Analysing interests... (may take some time)"):
            shared_interests = self.analyse_shared_interests(handle)
            visualize_shared_interests(shared_interests)
