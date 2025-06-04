"""
Streamlit web application for facial keypoints detection
"""

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.cnn_model import create_model
from utils.inference import KeypointsPredictor
from utils.visualization import plot_keypoints_on_image, KEYPOINT_NAMES


# Page configuration
st.set_page_config(
    page_title="顔特徴点検出 / Facial Keypoints Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-box {
        background-color: #e1f5fe;
        padding: 0.5rem;
        border-radius: 0.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def find_latest_model():
    """Find the most recently trained model."""
    import glob
    import os
    
    # Search for model files in common locations
    search_paths = [
        "checkpoints/*.pth",
        "checkpoints/*.pt", 
        "models/*.pth",
        "models/*.pt",
        "*.pth",
        "*.pt",
        "../checkpoints/*.pth",
        "../checkpoints/*.pt",
        "../*.pth",
        "../*.pt"
    ]
    
    model_files = []
    for pattern in search_paths:
        model_files.extend(glob.glob(pattern))
    
    if not model_files:
        return None
    
    # Return the most recently modified model
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

@st.cache_resource
def load_model_automatically():
    """Load and cache the latest available model."""
    try:
        # Find latest model
        model_path = find_latest_model()
        if not model_path:
            return None, None, "No trained model found"
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Try different model types until one works
        model_types = ['basic_cnn', 'deep_cnn', 'resnet18', 'resnet34', 'efficientnet_b0']
        
        for model_type in model_types:
            try:
                # Create model
                model = create_model(model_type=model_type, num_keypoints=30)
                
                # Create predictor
                predictor = KeypointsPredictor(
                    model=model,
                    model_path=model_path,
                    device=device,
                    image_size=(96, 96)
                )
                
                return predictor, device, f"Model loaded: {model_type} from {os.path.basename(model_path)}"
            except Exception as e:
                continue
        
        return None, None, f"Failed to load model from {model_path}"
        
    except Exception as e:
        return None, None, f"Error loading model: {e}"


def process_uploaded_image(uploaded_file):
    """Process uploaded image file."""
    try:
        # Read image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        return image_np, image
    except Exception as e:
        st.error(f"画像の処理に失敗しました / Failed to process image: {e}")
        return None, None


def predict_keypoints(predictor, image_np):
    """Predict keypoints for an image."""
    try:
        # Predict keypoints
        keypoints, confidence = predictor.predict(image_np, return_confidence=True)
        
        return keypoints, confidence
    except Exception as e:
        st.error(f"特徴点の検出に失敗しました / Failed to detect keypoints: {e}")
        return None, None


def create_keypoints_visualization(image_np, keypoints):
    """Create visualization of keypoints on image."""
    try:
        # Create plot
        fig = plot_keypoints_on_image(
            image=cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY),
            keypoints=keypoints,
            title="検出された顔特徴点 / Detected Facial Keypoints",
            denormalize=False,
            image_size=image_np.shape[:2],
            show_labels=False
        )
        
        return fig
    except Exception as e:
        st.error(f"可視化の作成に失敗しました / Failed to create visualization: {e}")
        return None


def create_annotated_image(image_np, keypoints, language="日本語"):
    """Create an annotated image with keypoints and labels overlaid."""
    try:
        # Copy the original image
        annotated_image = image_np.copy()
        
        # Reshape keypoints to (15, 2) if needed
        if keypoints.shape[0] == 30:
            keypoints_2d = keypoints.reshape(15, 2)
        else:
            keypoints_2d = keypoints
        
        # Define keypoint labels in Japanese and English
        keypoint_labels = {
            "日本語": [
                "左目中心", "右目中心", "左目内角", "左目外角",
                "右目内角", "右目外角", "左眉内端", "左眉外端",
                "右眉内端", "右眉外端", "鼻先", "口左角",
                "口右角", "上唇中央", "下唇中央"
            ],
            "English": [
                "L Eye Center", "R Eye Center", "L Eye Inner", "L Eye Outer",
                "R Eye Inner", "R Eye Outer", "L Brow Inner", "L Brow Outer",
                "R Brow Inner", "R Brow Outer", "Nose Tip", "Mouth L",
                "Mouth R", "Upper Lip", "Lower Lip"
            ]
        }
        
        # Define colors for different features (BGR format for OpenCV)
        feature_colors = {
            'eyes': (0, 255, 0),        # Green
            'eyebrows': (0, 0, 255),    # Red
            'nose': (255, 0, 0),        # Blue
            'mouth': (0, 255, 255)      # Yellow
        }
        
        # Define feature groups
        feature_groups = {
            'eyes': [0, 1, 2, 3, 4, 5],      # Eyes
            'eyebrows': [6, 7, 8, 9],        # Eyebrows
            'nose': [10],                     # Nose
            'mouth': [11, 12, 13, 14]        # Mouth
        }
        
        # Get image dimensions
        h, w = image_np.shape[:2]
        
        # Draw keypoints and labels
        for i, (x, y) in enumerate(keypoints_2d):
            # Convert normalized coordinates to pixel coordinates
            pixel_x = int(x * w)
            pixel_y = int(y * h)
            
            # Skip if coordinates are invalid
            if pixel_x < 0 or pixel_y < 0 or pixel_x >= w or pixel_y >= h:
                continue
                
            # Determine color based on feature group
            color = (255, 255, 255)  # Default white
            for feature, indices in feature_groups.items():
                if i in indices:
                    color = feature_colors[feature]
                    break
            
            # Draw circle marker
            cv2.circle(annotated_image, (pixel_x, pixel_y), 5, color, -1)
            cv2.circle(annotated_image, (pixel_x, pixel_y), 6, (0, 0, 0), 2)  # Black border
            
            # Add label
            label = keypoint_labels[language][i]
            
            # Calculate text position (offset to avoid overlapping with point)
            text_x = pixel_x + 10
            text_y = pixel_y - 10
            
            # Ensure text is within image bounds
            if text_x + len(label) * 6 > w:
                text_x = pixel_x - len(label) * 6 - 10
            if text_y < 20:
                text_y = pixel_y + 25
                
            # Draw text background for better visibility
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated_image, (text_x - 2, text_y - text_h - 4), 
                         (text_x + text_w + 2, text_y + 4), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(annotated_image, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated_image
        
    except Exception as e:
        st.error(f"注釈付き画像の作成に失敗しました / Failed to create annotated image: {e}")
        return None


def display_keypoints_table(keypoints):
    """Display keypoints in a table format."""
    # Reshape keypoints to (15, 2)
    keypoints_2d = keypoints.reshape(15, 2)
    
    # Create table data
    table_data = []
    for i, (x, y) in enumerate(keypoints_2d):
        point_name = KEYPOINT_NAMES[i*2].replace('_x', '').replace('_', ' ').title()
        table_data.append({
            'Point': point_name,
            'X Coordinate': f"{x:.2f}",
            'Y Coordinate': f"{y:.2f}"
        })
    
    return table_data


def main():
    """Main application function."""
    # Title
    st.markdown('<h1 class="main-header">👁️ 顔特徴点検出システム / Facial Keypoints Detection System</h1>', 
                unsafe_allow_html=True)
    
    # Language selection
    language = st.sidebar.selectbox(
        "言語選択 / Language Selection",
        ["日本語", "English"],
        index=0
    )
    
    # Texts based on language
    if language == "日本語":
        texts = {
            'model_status': 'モデル状態',
            'image_upload': '画像アップロード',
            'upload_instruction': 'JPEGまたはPNG画像をアップロードしてください',
            'image_requirements': '画像要件',
            'req_format': '対応形式: JPEG, PNG',
            'req_size': '推奨サイズ: 96x96ピクセル以上',
            'req_face': '顔が中心に位置する正面向きの画像が最適',
            'req_single': '単一の顔が含まれる画像を推奨',
            'predict_button': '特徴点を検出',
            'results': '検出結果',
            'confidence': '信頼度',
            'coordinates': '座標値',
            'visualization': '可視化結果',
            'no_model': 'モデルが読み込まれていません',
            'no_image': '画像をアップロードしてください',
            'processing': '処理中...'
        }
    else:
        texts = {
            'model_status': 'Model Status',
            'image_upload': 'Image Upload',
            'upload_instruction': 'Please upload a JPEG or PNG image',
            'image_requirements': 'Image Requirements',
            'req_format': 'Supported formats: JPEG, PNG',
            'req_size': 'Recommended size: 96x96 pixels or larger',
            'req_face': 'Front-facing images with face centered work best',
            'req_single': 'Single face per image recommended',
            'predict_button': 'Detect Keypoints',
            'results': 'Detection Results',
            'confidence': 'Confidence',
            'coordinates': 'Coordinates',
            'visualization': 'Visualization',
            'no_model': 'No model loaded',
            'no_image': 'Please upload an image',
            'processing': 'Processing...'
        }
    
    # Sidebar - Model status
    st.sidebar.header(texts['model_status'])
    
    # Load model automatically
    predictor, device, status_message = load_model_automatically()
    
    if predictor is not None:
        st.sidebar.success(f"✅ {status_message}")
        st.sidebar.info(f"Device: {device}")
    else:
        st.sidebar.error(f"❌ {status_message}")
        st.sidebar.info("Please ensure you have trained a model first.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Image upload section
        st.header(texts['image_upload'])
        
        uploaded_file = st.file_uploader(
            texts['upload_instruction'],
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image for facial keypoints detection"
        )
        
        # Image requirements info
        with st.expander(texts['image_requirements']):
            st.markdown(f"""
            - {texts['req_format']}
            - {texts['req_size']}
            - {texts['req_face']}
            - {texts['req_single']}
            """)
        
        # Display uploaded image
        if uploaded_file is not None:
            image_np, image_pil = process_uploaded_image(uploaded_file)
            
            if image_np is not None:
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
                
                # Image info
                st.markdown(f"""
                <div class="info-box">
                <strong>Image Info:</strong><br>
                Size: {image_np.shape[1]} x {image_np.shape[0]} pixels<br>
                Channels: {image_np.shape[2] if len(image_np.shape) > 2 else 1}
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Results section
        st.header(texts['results'])
        
        if uploaded_file is not None and predictor is not None:
            # Predict button
            if st.button(texts['predict_button'], type="primary"):
                with st.spinner(texts['processing']):
                    # Predict keypoints
                    keypoints, confidence = predict_keypoints(predictor, image_np)
                    
                    if keypoints is not None:
                        # Display confidence
                        st.markdown(f"""
                        <div class="metric-box">
                        <h4>{texts['confidence']}: {confidence:.3f}</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create and display annotated image
                        annotated_image = create_annotated_image(image_np, keypoints, language)
                        if annotated_image is not None:
                            st.subheader(texts['visualization'])
                            st.image(annotated_image, caption="検出された顔特徴点 / Detected Facial Keypoints", 
                                   use_column_width=True)
                        
                        # Also create and display the original matplotlib visualization
                        fig = create_keypoints_visualization(image_np, keypoints)
                        if fig is not None:
                            st.pyplot(fig)
                            plt.close(fig)  # Close to free memory
                        
                        # Display coordinates table
                        st.subheader(texts['coordinates'])
                        table_data = display_keypoints_table(keypoints)
                        st.dataframe(table_data, use_container_width=True)
                        
                        # Download results
                        results_data = {
                            'keypoints': keypoints.tolist(),
                            'confidence': confidence,
                            'image_shape': image_np.shape
                        }
                        
                        st.download_button(
                            label="Download Results (JSON)",
                            data=str(results_data),
                            file_name="keypoints_results.json",
                            mime="application/json"
                        )
        
        elif predictor is None:
            st.warning(texts['no_model'])
        else:
            st.info(texts['no_image'])
    
    # Footer with additional info
    st.markdown("---")
    
    with st.expander("About This Application / このアプリについて"):
        if language == "日本語":
            st.markdown("""
            ## 顔特徴点検出システム
            
            このアプリケーションは、深層学習を使用して顔画像から15個の特徴点を検出します。
            
            **検出される特徴点:**
            - 左右の目の中心と角
            - 左右の眉毛の端
            - 鼻の先端
            - 口の角と上下唇の中心
            
            **使用技術:**
            - PyTorch (深層学習フレームワーク)
            - Streamlit (Webアプリフレームワーク)
            - OpenCV (画像処理)
            - ClearML (実験管理)
            """)
        else:
            st.markdown("""
            ## Facial Keypoints Detection System
            
            This application uses deep learning to detect 15 facial keypoints from face images.
            
            **Detected Keypoints:**
            - Left and right eye centers and corners
            - Left and right eyebrow ends
            - Nose tip
            - Mouth corners and center of upper/lower lips
            
            **Technologies Used:**
            - PyTorch (Deep Learning Framework)
            - Streamlit (Web App Framework)
            - OpenCV (Image Processing)
            - ClearML (Experiment Management)
            """)


if __name__ == "__main__":
    main()