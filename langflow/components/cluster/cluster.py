import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from datetime import datetime
from langflow.custom import Component
from langflow.inputs import Input, IntInput
from langflow.schema.message import Message
from langflow.schema.content_block import ContentBlock
from langflow.schema.content_types import TextContent, MediaContent
from langflow.io import Output

class CustomComponent(Component):
    display_name = "CSV Data Clusterer"
    description = "Cluster data based on CSV columns (not document text)."
    icon = "custom_components"
    name = "CSVDataClusterer"
    inputs = [
        Input(
            name="embeddings_data",
            display_name="Embeddings Data",
            required=True,
            input_types=["Data"]
        ),
        Input(
            name="csv_dataframe",
            display_name="CSV DataFrame",
            required=True,
            input_types=["DataFrame", "pandas"]
        ),
        Input(
            name="label_column",
            display_name="Label Column (for analysis)",
            field_type="str",
            required=False,
        ),
        IntInput(
            name="num_clusters",
            display_name="Number of Clusters",
            required=True,
            value=3,
        ),
        IntInput(
            name="perplexity",
            display_name="t-SNE Perplexity",
            required=False,
            value=30,
        ),
    ]
    outputs = [
        Output(display_name="Output Message", name="output_message", method="cluster_csv_data"),
    ]
    
    def extract_dataframe(self):
        """Extract DataFrame from input - expecting proper DataFrame"""
        return self.csv_dataframe
    
    def extract_embeddings(self):
        """Extract embeddings and ensure they match DataFrame rows"""
        embeddings_data = self.embeddings_data
        df = self.csv_dataframe
        
        print(f"[DEBUG] DataFrame shape: {df.shape}")
        
        if hasattr(embeddings_data, 'data') and isinstance(embeddings_data.data, dict):
            if 'embeddings' in embeddings_data.data:
                emb_raw = embeddings_data.data['embeddings']
                print(f"[DEBUG] Raw embedding length: {len(emb_raw)}")
                
                # Convert to numpy array
                embeddings = np.array(emb_raw)
                
                # The embeddings should represent the CSV rows
                # If we have one long vector but multiple CSV rows, we need to handle this
                num_csv_rows = len(df)
                
                if len(embeddings.shape) == 1:  # Single flattened vector
                    if len(embeddings) == num_csv_rows:
                        # Each embedding value corresponds to one CSV row (1D embedding per row)
                        embeddings = embeddings.reshape(-1, 1)
                        print(f"[DEBUG] Using 1D embeddings: {embeddings.shape}")
                    elif len(embeddings) % num_csv_rows == 0:
                        # Split the embedding into equal parts for each row
                        embedding_dim = len(embeddings) // num_csv_rows
                        embeddings = embeddings.reshape(num_csv_rows, embedding_dim)
                        print(f"[DEBUG] Split embedding: {num_csv_rows} rows × {embedding_dim} dims")
                    else:
                        # Use the full embedding as a single high-dim vector per row
                        # Replicate for each row (not ideal but handles the mismatch)
                        embeddings = np.tile(embeddings.reshape(1, -1), (num_csv_rows, 1))
                        print(f"[DEBUG] Replicated embedding for each row: {embeddings.shape}")
                
                if len(embeddings) != num_csv_rows:
                    raise ValueError(f"Embedding count ({len(embeddings)}) doesn't match CSV rows ({num_csv_rows})")
                
                print(f"[DEBUG] Final embeddings shape: {embeddings.shape}")
                return embeddings
            else:
                raise ValueError("No 'embeddings' key found in data")
        else:
            raise ValueError("Could not extract embeddings from data")
    
    def generate_cluster_label_with_llm(self, cluster_data, cluster_id):
        """Generate intelligent cluster labels using LLM based on CSV data patterns"""
        llm_component = self.llm
        
        # Get the LLM model
        if hasattr(llm_component, 'build_model'):
            llm_model = llm_component.build_model()
        else:
            llm_model = llm_component
        
        # Analyze the cluster data to create a summary
        patterns = []
        
        # Analyze each column in the cluster
        for col in cluster_data.columns[:8]:  # Limit to first 8 columns
            if cluster_data[col].dtype in ['object', 'category']:
                # For categorical columns, get most common values
                top_vals = cluster_data[col].value_counts().head(3)
                if not top_vals.empty:
                    values = [f"{v}({c})" for v, c in top_vals.items()]
                    patterns.append(f"{col}: {', '.join(values)}")
            else:
                # For numerical columns, get statistics
                try:
                    mean_val = cluster_data[col].mean()
                    min_val = cluster_data[col].min()
                    max_val = cluster_data[col].max()
                    if not pd.isna(mean_val):
                        patterns.append(f"{col}: avg={mean_val:.1f}, range={min_val:.1f}-{max_val:.1f}")
                except:
                    pass
        
        # Create prompt for LLM
        cluster_size = len(cluster_data)
        patterns_text = "\n".join(patterns[:6])  # Limit patterns to avoid long prompts
        
        prompt = f"""Analyze this data cluster and create a short, descriptive label (2-4 words).

Cluster {cluster_id} contains {cluster_size} data points with these patterns:
{patterns_text}

Based on these data patterns, what would be a concise, meaningful label for this group?
Respond with ONLY the label, no explanations."""
        
        try:
            # Use the LLM to generate label
            if hasattr(llm_model, 'invoke'):
                response = llm_model.invoke(prompt)
                label = getattr(response, 'content', str(response))
            else:
                response = llm_model(prompt)
                label = getattr(response, 'content', str(response))
            
            # Clean up the label
            label = label.strip().strip('"\'').strip('.')
            print(f"[DEBUG] Generated label for cluster {cluster_id}: {label}")
            return label
        except Exception as e:
            print(f"[DEBUG] Error generating label for cluster {cluster_id}: {str(e)}")
            # Fallback to simple label
            return f"Group {cluster_id}"
        """Create image URL"""
        base_url = os.environ.get('LANGFLOW_BASE_URL', '').rstrip('/')
        flow_id = getattr(self, 'flow_id', 'default_flow')
        return f"{base_url}/api/v1/files/images/{flow_id}/{filename}"
    
    def cluster_csv_data(self) -> Message:
        """Cluster CSV data using provided embeddings"""
        try:
            # Extract DataFrame and embeddings
            df = self.extract_dataframe()
            embeddings = self.extract_embeddings()
            
            # Ensure embeddings and data match
            if len(embeddings) != len(df):
                return Message(text=f"Error: Embeddings ({len(embeddings)}) and data rows ({len(df)}) count mismatch", sender="csv_clusterer")
            
            # Perform clustering using embeddings
            n_clusters = min(self.num_clusters, len(df))
            if n_clusters < 1:
                n_clusters = 1
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(embeddings)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            if len(embeddings) == 1:
                # Single data point
                plt.scatter([0], [0], s=100, label="Single Data Point")
                plt.title('Single Data Point')
            elif len(embeddings) >= 2:
                # Use t-SNE for visualization
                perplexity_value = getattr(self, 'perplexity', 30)
                perplexity = max(1, min(perplexity_value, len(embeddings) - 1))
                
                if embeddings.shape[1] == 1:
                    # 1D embeddings - plot on x-axis
                    coords = np.column_stack([embeddings.flatten(), np.zeros(len(embeddings))])
                elif embeddings.shape[1] == 2:
                    # 2D embeddings - use directly
                    coords = embeddings
                else:
                    # Multi-dimensional embeddings - use t-SNE
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    coords = tsne.fit_transform(embeddings)
                
                # Plot clusters
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                
                for i in range(n_clusters):
                    mask = clusters == i
                    color = colors[i % len(colors)]
                    plt.scatter(coords[mask, 0], coords[mask, 1], 
                              c=color, label=f"Cluster {i} ({np.sum(mask)})", alpha=0.7, s=50)
                
                plt.title(f'Data Clustering Using Embeddings (dims: {embeddings.shape[1]})')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.legend()
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot
            root_dir = os.environ.get('LANGFLOW_CONFIG_DIR', '.')
            flow_id = getattr(self, 'flow_id', 'default_flow')
            output_dir = os.path.join(root_dir, flow_id)
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"embedding_clusters_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Add cluster assignments to DataFrame
            df_results = df.copy()
            df_results['cluster'] = clusters
            df_results['cluster_label'] = [cluster_labels.get(c, f'Cluster_{c}') for c in clusters]
            
            # Save results CSV
            csv_filename = f"embedding_clustered_data_{timestamp}.csv"
            csv_filepath = os.path.join(output_dir, csv_filename)
            df_results.to_csv(csv_filepath, index=False)
            
            # Create summary
            summary = f"Data clustering using embeddings (dimension: {embeddings.shape[1]})\n\n"
            summary += f"Dataset: {len(df)} rows clustered into {n_clusters} groups\n\n"
            
            for i in range(n_clusters):
                count = np.sum(clusters == i)
                summary += f"**Cluster {i}**: {count} rows\n"
                
                # Show some data examples for this cluster
                cluster_data = df[clusters == i]
                if len(cluster_data) > 0:
                    # Show first few columns as examples
                    for col in df.columns[:3]:  # Show first 3 columns
                        if df[col].dtype in ['object', 'category']:
                            # Most common values for categorical
                            top_vals = cluster_data[col].value_counts().head(2)
                            if not top_vals.empty:
                                summary += f"  {col}: {', '.join([f'{v}({c})' for v, c in top_vals.items()])}\n"
                        else:
                            # Stats for numerical
                            try:
                                mean_val = cluster_data[col].mean()
                                summary += f"  {col}: avg={mean_val:.2f}\n"
                            except:
                                pass
                summary += "\n"
            
            # Add label column analysis if specified
            if hasattr(self, 'label_column') and self.label_column and self.label_column in df.columns:
                summary += f"\nLabel distribution by cluster ({self.label_column}):\n"
                for i in range(n_clusters):
                    cluster_data = df[clusters == i]
                    label_dist = cluster_data[self.label_column].value_counts().head(3)
                    if not label_dist.empty:
                        summary += f"Cluster {i}: {', '.join([f'{v}({c})' for v, c in label_dist.items()])}\n"
            
            image_url = self.create_image_url(filename)
            
            return Message(
                text=f"Embedding-based clustering completed: {len(df)} rows → {n_clusters} clusters\n\nEmbedding dims: {embeddings.shape[1]}\n\nVisualization: {image_url}",
                sender="csv_clusterer",
                sender_name="CSV Data Clusterer",
                content_blocks=[
                    ContentBlock(
                        title="Embedding-Based Clustering Visualization",
                        contents=[MediaContent(type="media", urls=[image_url])]
                    ),
                    ContentBlock(
                        title="Cluster Analysis",
                        contents=[TextContent(type="text", text=summary)]
                    )
                ],
            )
            
        except Exception as e:
            return Message(text=f"Error: {str(e)}", sender="csv_clusterer")