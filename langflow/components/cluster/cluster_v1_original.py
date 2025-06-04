import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import argparse
import os
import requests
import json
import time
import re
from typing import List, Dict, Tuple, Optional

def load_data(file_path: str, text_column: Optional[str] = None, additional_columns: Optional[List[str]] = None) -> Tuple[List[str], pd.DataFrame]:
    """Load text data from CSV file with support for multiple columns"""
    df = pd.read_csv(file_path)
    full_df = df.copy()
    
    # Use specified text column or default to first column
    if text_column is not None and text_column in df.columns:
        texts = df[text_column].fillna('').astype(str).tolist()
    else:
        text_column = df.columns[0]
        texts = df[text_column].fillna('').astype(str).tolist()
        print(f"Using '{text_column}' as the primary text column")
    
    # If additional columns are specified, combine them with the main text
    if additional_columns:
        valid_columns = [col for col in additional_columns if col in df.columns]
        
        if valid_columns:
            print(f"Including additional columns: {', '.join(valid_columns)}")
            
            # Create combined texts
            combined_texts = []
            for i, main_text in enumerate(texts):
                parts = [main_text]
                
                # Add content from additional columns
                for col in valid_columns:
                    if pd.notna(df[col].iloc[i]):
                        col_value = str(df[col].iloc[i]).strip()
                        if col_value:
                            parts.append(f"{col}: {col_value}")
                
                combined_texts.append(" ".join(parts))
            
            texts = combined_texts
    
    # Remove empty strings
    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    filtered_texts = [texts[i] for i in valid_indices]
    
    # Filter the full dataframe to match the filtered texts
    full_df = full_df.iloc[valid_indices].reset_index(drop=True)
    
    return filtered_texts, full_df

def get_embeddings_with_remote_api(
    texts: List[str], 
    embedding_api_url: str, 
    api_key: str = "Bearer None", 
    model_name: str = "text-embedding-ada-002",
    batch_size: int = 20
) -> np.ndarray:
    """Generate embeddings with a remote API"""
    print(f"Generating embeddings via API using {model_name} model...")
    
    # Set up headers with Bearer token auth
    headers = {"Content-Type": "application/json"}
    if api_key and api_key.lower() != "none":
        if api_key.startswith("Bearer "):
            headers["Authorization"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    
    # Process in batches
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Prepare API request
        data = {
            "model": model_name,
            "input": batch
        }
        
        # Make API call
        try:
            response = requests.post(embedding_api_url, headers=headers, json=data, timeout=90)
            
            if response.status_code == 200:
                result = response.json()
                api_embeddings = None
                
                # Try to extract embeddings from different API response formats
                if "data" in result and isinstance(result["data"], list):
                    try:
                        api_embeddings = [item["embedding"] for item in result["data"]]
                    except:
                        pass
                
                if api_embeddings is None and "embeddings" in result:
                    api_embeddings = result["embeddings"]
                
                if api_embeddings is None:
                    # Look for any list of lists matching the expected count
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) == len(batch):
                            if all(isinstance(item, list) for item in value):
                                api_embeddings = value
                                break
                
                if api_embeddings:
                    all_embeddings.extend([np.array(emb) for emb in api_embeddings])
                else:
                    raise Exception("Could not extract embeddings from API response")
            else:
                raise Exception(f"API call failed: {response.status_code}")
        except Exception as e:
            print(f"Error getting embeddings: {str(e)}")
            # Provide simple fallback embeddings in case of failure
            dim = 1536  # Typical embedding dimension
            for _ in range(len(batch)):
                all_embeddings.append(np.random.normal(0, 0.1, dim))
    
    # Convert to numpy array
    embeddings = np.vstack(all_embeddings)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return embeddings

def generate_label_via_api(
    texts: List[str], 
    terms: List[str], 
    llm_api_base: str, 
    llm_api_key: str = "None", 
    api_model: str = "gpt-3.5-turbo"
) -> str:
    """Generate a label using an LLM API"""
    chat_endpoint = f"{llm_api_base.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    # Add API key to headers if provided
    if llm_api_key and llm_api_key.lower() != "none":
        headers["Authorization"] = f"Bearer {llm_api_key}"
    
    # Create a sample of texts
    sample_texts = texts[:3]  # Take up to 3 examples
    sample_texts = [t[:150] + "..." if len(t) > 150 else t for t in sample_texts]
    
    prompt = f"""
Create a short, descriptive label (3-5 words) for a group of documents with these common themes.
Key terms: {', '.join(terms[:10])}
Example documents:
1. {sample_texts[0] if len(sample_texts) > 0 else ""}
2. {sample_texts[1] if len(sample_texts) > 1 else ""}
3. {sample_texts[2] if len(sample_texts) > 2 else ""}
Respond with ONLY the label, no explanations.
"""
    
    data = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": "You create concise, descriptive labels for document clusters."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 20,
        "temperature": 0.2
    }
    
    try:
        response = requests.post(chat_endpoint, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            label = result["choices"][0]["message"]["content"].strip()
            return label.strip('"\'').strip('.')
    except:
        pass
    
    # Fallback to using top terms
    if terms:
        return f"{terms[0].title()} & {terms[1].title()}" if len(terms) > 1 else terms[0].title()
    return "Unlabeled Cluster"

def generate_profile_based_label(
    cluster_texts: List[str],
    cluster_df: pd.DataFrame,
    profile_columns: List[str],
    llm_api_base: str,
    llm_api_key: str,
    api_model: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate a cluster label based on profile attributes using LLM
    
    Args:
        cluster_texts: List of text documents in this cluster
        cluster_df: DataFrame containing data for this cluster
        profile_columns: List of columns containing profile attributes
        llm_api_base: Base URL for LLM API
        llm_api_key: API key for LLM
        api_model: Model name for labeling
        
    Returns:
        A descriptive label string
    """
    # Extract profile attribute information
    profile_info = []
    
    for col in profile_columns:
        if col in cluster_df.columns:
            if cluster_df[col].dtype == 'object':
                # For text columns, get most common values
                try:
                    top_values = cluster_df[col].value_counts().head(3)
                    if not top_values.empty:
                        values_str = ", ".join([f"{v} ({c})" for v, c in zip(top_values.index, top_values.values)])
                        profile_info.append(f"{col}: {values_str}")
                except:
                    pass
            else:
                # For numeric columns, get stats
                try:
                    avg = cluster_df[col].mean()
                    profile_info.append(f"{col}: avg={avg:.2f}")
                except:
                    pass
    
    # Extract key terms from texts
    try:
        vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
        X = vectorizer.fit_transform(cluster_texts)
        feature_names = vectorizer.get_feature_names_out()
        importance = np.asarray(X.sum(axis=0)).flatten()
        top_indices = importance.argsort()[-10:][::-1]
        top_terms = [feature_names[i] for i in top_indices]
        
        profile_info.append(f"Key terms: {', '.join(top_terms)}")
    except:
        top_terms = []
    
    # Fallback label in case LLM fails
    fallback_label = "Unlabeled Cluster"
    if top_terms:
        fallback_label = f"{top_terms[0].title()} & {top_terms[1].title()}" if len(top_terms) > 1 else top_terms[0].title()
    
    # Sample some texts
    sample_texts = cluster_texts[:2]
    sample_texts = [t[:150] + "..." if len(t) > 150 else t for t in sample_texts]
    
    # Create prompt for LLM
    profile_details = "\n".join(profile_info)
    prompt = f"""
Create a short, descriptive label (3-5 words) for a group of professionals with these attributes:

{profile_details}

Sample texts:
1. {sample_texts[0] if sample_texts else ""}
2. {sample_texts[1] if len(sample_texts) > 1 else ""}

The label should focus on what unites these professionals (role, function, domain, etc).
Respond with ONLY the label, no explanations.
"""
    
    # Call LLM API
    try:
        chat_endpoint = f"{llm_api_base.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {llm_api_key}"
        }
        
        data = {
            "model": api_model,
            "messages": [
                {"role": "system", "content": "You create concise labels for professional groups."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 20,
            "temperature": 0.2
        }
        
        response = requests.post(chat_endpoint, headers=headers, json=data, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            label = result["choices"][0]["message"]["content"].strip()
            return label.strip('"\'').strip('.')
    except:
        pass
    
    return fallback_label

def cluster_and_label(
    texts: List[str], 
    embeddings: np.ndarray, 
    n_clusters: int, 
    llm_api_base: str, 
    llm_api_key: str = "None", 
    api_model: str = "gpt-3.5-turbo"
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Perform clustering and generate labels"""
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Extract top terms for each cluster to help with labeling
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    cluster_labels = {}
    
    for cluster_id in range(n_clusters):
        # Get texts in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_texts = [texts[i] for i in cluster_indices]
        
        if not cluster_texts:
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Get top terms
        try:
            X = vectorizer.fit_transform(cluster_texts)
            feature_names = vectorizer.get_feature_names_out()
            importance = np.asarray(X.sum(axis=0)).flatten()
            top_indices = importance.argsort()[-10:][::-1]
            top_terms = [feature_names[i] for i in top_indices]
        except:
            top_terms = []
        
        # Generate label using LLM API
        try:
            label = generate_label_via_api(cluster_texts, top_terms, llm_api_base, llm_api_key, api_model)
            cluster_labels[cluster_id] = label
            print(f"Cluster {cluster_id}: {label} ({len(cluster_texts)} items)")
        except:
            # Fallback to simple labeling
            if top_terms:
                fallback_label = f"{top_terms[0].title()} & {top_terms[1].title()}" if len(top_terms) > 1 else top_terms[0].title()
                cluster_labels[cluster_id] = fallback_label
            else:
                cluster_labels[cluster_id] = f"Cluster {cluster_id}"
    
    return clusters, cluster_labels

def cluster_and_label_profiles(
    texts: List[str],
    embeddings: np.ndarray,
    df_original: pd.DataFrame,
    n_clusters: int,
    profile_columns: List[str],
    llm_api_base: str,
    llm_api_key: str,
    api_model: str = "gpt-3.5-turbo"
) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Perform clustering and generate profile-aware labels
    
    Args:
        texts: List of text documents
        embeddings: Document embeddings
        df_original: Original dataframe with all columns
        n_clusters: Number of clusters to create
        profile_columns: List of columns containing profile attributes
        llm_api_base: Base URL for LLM API
        llm_api_key: API key for LLM
        api_model: Model name for labeling
        
    Returns:
        Tuple of (cluster assignments, cluster labels)
    """
    # Cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Generate labels for each cluster
    cluster_labels = {}
    
    for cluster_id in range(n_clusters):
        # Get items in this cluster
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_texts = [texts[i] for i in cluster_indices]
        cluster_df = df_original.iloc[cluster_indices].reset_index(drop=True)
        
        if not cluster_texts:
            print(f"Warning: Cluster {cluster_id} is empty")
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Generate profile-based label
        label = generate_profile_based_label(
            cluster_texts,
            cluster_df,
            profile_columns,
            llm_api_base,
            llm_api_key,
            api_model
        )
        
        cluster_labels[cluster_id] = label
        print(f"Cluster {cluster_id}: {label} ({len(cluster_texts)} items)")
    
    return clusters, cluster_labels

def analyze_dataset(df: pd.DataFrame, text_column: Optional[str] = None) -> Dict[str, str]:
    """Generate basic insights about the dataset without LLM"""
    insights = {
        "dataset_overview": "",
        "key_patterns": "",
        "suggested_analysis": ""
    }
    
    try:
        # Dataset overview
        row_count = len(df)
        col_count = len(df.columns)
        insights["dataset_overview"] = f"Dataset with {row_count} rows and {col_count} columns."
        
        # Basic patterns
        missing_data = df.isnull().sum()
        cols_with_missing = missing_data[missing_data > 0]
        if not cols_with_missing.empty:
            insights["key_patterns"] = f"Found missing data in {len(cols_with_missing)} columns."
        else:
            insights["key_patterns"] = "No missing values found."
            
        # Analysis suggestions
        suggestions = [
            "Consider exploring relationships between features",
            "Look for patterns across different clusters",
            "Analyze how document features correlate with cluster assignments"
        ]
        insights["suggested_analysis"] = " ".join(suggestions)
    except:
        pass
    
    return insights

def visualize_clusters(
    embeddings: np.ndarray, 
    clusters: np.ndarray, 
    cluster_labels: Dict[int, str],
    df_original: pd.DataFrame,
    output_path: str = "cluster_visualization.png",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    normalize_axes: bool = False,
    figsize: Tuple[int, int] = (12, 10),
    x_dimension: int = 0,
    y_dimension: int = 1,
    x_column: Optional[str] = None,
    y_column: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a visualization of the clusters
    
    Args:
        embeddings: Document embeddings
        clusters: Cluster assignments
        cluster_labels: Labels for each cluster
        df_original: Original dataframe with all columns
        output_path: Where to save the visualization
        x_range: Optional tuple of (min, max) for x-axis
        y_range: Optional tuple of (min, max) for y-axis
        normalize_axes: Whether to normalize axes to range [-1, 1]
        figsize: Size of the figure (width, height) in inches
        x_dimension: Which t-SNE dimension to use for X axis (0 or 1), used if x_column is None
        y_dimension: Which t-SNE dimension to use for Y axis (0 or 1), used if y_column is None
        x_column: Name of dataframe column to use for X axis (overrides x_dimension)
        y_column: Name of dataframe column to use for Y axis (overrides y_dimension)
    
    Returns:
        DataFrame with visualization data
    """
    # First, generate t-SNE coordinates as a fallback
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    reduced_embeddings = tsne.fit_transform(embeddings)
    
    # Normalize t-SNE to [-1, 1] range if requested
    if normalize_axes:
        for col in range(reduced_embeddings.shape[1]):
            col_data = reduced_embeddings[:, col]
            col_max = np.max(np.abs(col_data))
            if col_max > 0:
                reduced_embeddings[:, col] = reduced_embeddings[:, col] / col_max
    
    # Ensure dimensions are valid (0 or 1)
    x_dimension = max(0, min(1, x_dimension))
    y_dimension = max(0, min(1, y_dimension))
    
    # Initialize data for plotting
    df_viz = pd.DataFrame({'cluster': clusters})
    
    # Determine X coordinate values and label
    if x_column and x_column in df_original.columns:
        # Use data column from original dataframe
        try:
            x_values = df_original[x_column].values
            x_label = x_column
            df_viz['x'] = x_values
            print(f"Using '{x_column}' column for X axis")
        except:
            # Fall back to t-SNE if there's an issue
            print(f"Warning: Problem using '{x_column}' for X axis. Falling back to t-SNE dimension {x_dimension}.")
            x_values = reduced_embeddings[:, x_dimension]
            x_label = f"Dimension {x_dimension}"
            df_viz['x'] = x_values
    else:
        # Use t-SNE dimension
        x_values = reduced_embeddings[:, x_dimension]
        x_label = f"Dimension {x_dimension}"
        df_viz['x'] = x_values
    
    # Determine Y coordinate values and label
    if y_column and y_column in df_original.columns:
        # Use data column from original dataframe
        try:
            y_values = df_original[y_column].values
            y_label = y_column
            df_viz['y'] = y_values
            print(f"Using '{y_column}' column for Y axis")
        except:
            # Fall back to t-SNE if there's an issue
            print(f"Warning: Problem using '{y_column}' for Y axis. Falling back to t-SNE dimension {y_dimension}.")
            y_values = reduced_embeddings[:, y_dimension]
            y_label = f"Dimension {y_dimension}"
            df_viz['y'] = y_values
    else:
        # Use t-SNE dimension
        y_values = reduced_embeddings[:, y_dimension]
        y_label = f"Dimension {y_dimension}"
        df_viz['y'] = y_values
    
    # Add cluster labels for display
    df_viz['label'] = df_viz['cluster'].map(lambda x: f"Cluster {x}: {cluster_labels.get(x, '')}")
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Get unique clusters
    unique_clusters = sorted(df_viz['cluster'].unique())
    
    # Generate distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_clusters))))
    
    for i, cluster_id in enumerate(unique_clusters):
        subset = df_viz[df_viz['cluster'] == cluster_id]
        if len(subset) > 0:  # Make sure subset is not empty
            plt.scatter(
                subset['x'], 
                subset['y'], 
                label=subset['label'].iloc[0],
                color=colors[i % len(colors)],
                alpha=0.7
            )
    
    # Set custom axis ranges if provided
    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)
    
    # Add grid for better readability
    plt.grid(alpha=0.3)
    
    plt.title('Document Clustering Visualization')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    return df_viz

def main() -> None:
    parser = argparse.ArgumentParser(description='Cluster documents using K-means and remote embeddings API')
    parser.add_argument('input_file', help='Path to input CSV file containing text data')
    parser.add_argument('--clusters', type=int, default=4, help='Number of clusters')
    parser.add_argument('--text-column', help='Name of the primary column containing text to cluster')
    parser.add_argument('--additional-columns', nargs='+', help='Additional columns to include in the analysis')
    parser.add_argument('--llm-api-base', required=True, help='Base URL for LLM API endpoints')
    parser.add_argument('--llm-api-key', default="None", help='API key for LLM API')
    parser.add_argument('--embedding-api-url', default="http://api-embed.apps.lumimai.com:8080/v1/embeddings", 
                        help='Full URL for embedding API endpoint')
    parser.add_argument('--embedding-api-key', default="Bearer None", 
                        help='API key for embedding API (default: "Bearer None")')
    parser.add_argument('--api-model', default="gpt-3.5-turbo", help='Model name for labeling')
    parser.add_argument('--embedding-model', default="text-embedding-ada-002", help='Model name for embeddings')
    parser.add_argument('--output', default='clustering_results', help='Prefix for output files')
    parser.add_argument('--batch-size', type=int, default=20, help='Batch size for API requests')
    
    # Visualization parameters
    parser.add_argument('--normalize-axes', action='store_true', help='Normalize visualization axes to [-1, 1] range')
    parser.add_argument('--x-min', type=float, help='Minimum value for X axis in visualization')
    parser.add_argument('--x-max', type=float, help='Maximum value for X axis in visualization')
    parser.add_argument('--y-min', type=float, help='Minimum value for Y axis in visualization') 
    parser.add_argument('--y-max', type=float, help='Maximum value for Y axis in visualization')
    parser.add_argument('--fig-width', type=int, default=12, help='Width of visualization figure in inches')
    parser.add_argument('--fig-height', type=int, default=10, help='Height of visualization figure in inches')
    parser.add_argument('--x-dimension', type=int, default=0, help='Which t-SNE dimension to use for X axis (0 or 1)')
    parser.add_argument('--y-dimension', type=int, default=1, help='Which t-SNE dimension to use for Y axis (0 or 1)')
    parser.add_argument('--x-column', help='Dataset column to use for X axis (overrides --x-dimension)')
    parser.add_argument('--y-column', help='Dataset column to use for Y axis (overrides --y-dimension)')
    
    args = parser.parse_args()
    
    # Start time for performance tracking
    start_time = time.time()
    
    try:
        # Load data with support for multiple columns
        texts, df_original = load_data(
            args.input_file, 
            text_column=args.text_column,
            additional_columns=args.additional_columns
        )
        print(f"Loaded {len(texts)} documents from {args.input_file}")
        
        # Basic dataset analysis
        insights = analyze_dataset(df_original, args.text_column)
        print(f"\nDataset Overview: {insights['dataset_overview']}")
        print(f"Key Patterns: {insights['key_patterns']}")
        
        # Generate embeddings via remote API
        embeddings = get_embeddings_with_remote_api(
            texts, 
            args.embedding_api_url,
            args.embedding_api_key,
            model_name=args.embedding_model,
            batch_size=args.batch_size
        )
        
        # Determine if we should use profile-aware clustering
        use_profile_labeling = (args.additional_columns and 
                               args.llm_api_key and 
                               args.llm_api_key.lower() != "none")
        
        if use_profile_labeling:
            print("\nUsing profile-aware clustering and labeling...")
            clusters, cluster_labels = cluster_and_label_profiles(
                texts,
                embeddings,
                df_original,
                args.clusters,
                args.additional_columns,
                args.llm_api_base,
                args.llm_api_key,
                args.api_model
            )
        else:
            print("\nUsing standard clustering and labeling...")
            clusters, cluster_labels = cluster_and_label(
                texts,
                embeddings,
                args.clusters,
                args.llm_api_base,
                args.llm_api_key,
                args.api_model
            )
        
        # Configure axis ranges if specified
        x_range = None
        if args.x_min is not None or args.x_max is not None:
            x_min = args.x_min if args.x_min is not None else -100
            x_max = args.x_max if args.x_max is not None else 100
            x_range = (x_min, x_max)
            
        y_range = None
        if args.y_min is not None or args.y_max is not None:
            y_min = args.y_min if args.y_min is not None else -100
            y_max = args.y_max if args.y_max is not None else 100
            y_range = (y_min, y_max)
            
        # Visualize with custom axes parameters
        df_viz = visualize_clusters(
            embeddings, 
            clusters, 
            cluster_labels,
            df_original,
            output_path=f"{args.output}_visualization.png",
            x_range=x_range,
            y_range=y_range,
            normalize_axes=args.normalize_axes,
            figsize=(args.fig_width, args.fig_height),
            x_dimension=args.x_dimension,
            y_dimension=args.y_dimension,
            x_column=args.x_column,
            y_column=args.y_column
        )
        
        # Save results - now including all original columns
        df_results = df_original.copy()
        df_results['cluster'] = clusters
        df_results['cluster_label'] = [f"{cluster_labels.get(c, '')}" for c in clusters]
        
        # Save the full results with all original data
        df_results.to_csv(f"{args.output}_results.csv", index=False)
        
        # Print summary
        print("\nClustering Summary:")
        for cluster_id, label in cluster_labels.items():
            cluster_size = np.sum(clusters == cluster_id)
            print(f"\nCluster {cluster_id}: {label} ({cluster_size} documents)")
            
            # Print feature analysis for this cluster if we have additional columns
            if args.additional_columns:
                print("  Feature analysis:")
                cluster_df = df_original.iloc[np.where(clusters == cluster_id)[0]]
                
                for col in args.additional_columns:
                    if col in df_original.columns:
                        # For categorical/text columns, show most common values
                        if df_original[col].dtype == 'object':
                            try:
                                top_values = cluster_df[col].value_counts().head(3)
                                if not top_values.empty:
                                    print(f"    {col}: {', '.join([f'{v} ({c})' for v, c in top_values.items()])}")
                            except:
                                pass
                        # For numeric columns, show average and range
                        elif pd.api.types.is_numeric_dtype(df_original[col]):
                            try:
                                avg = cluster_df[col].mean()
                                min_val = cluster_df[col].min()
                                max_val = cluster_df[col].max()
                                print(f"    {col}: avg={avg:.2f}, range={min_val:.2f}-{max_val:.2f}")
                            except:
                                pass
            
            # Print a few examples
            examples = [texts[i][:100] + "..." for i in np.where(clusters == cluster_id)[0][:3]]
            for i, example in enumerate(examples):
                print(f"  {i+1}. {example}")
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    main()