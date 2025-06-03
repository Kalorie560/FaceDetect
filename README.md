# 顔特徴点検出システム / Facial Keypoints Detection System

[English](#english) | [日本語](#japanese)

---

## English

### Overview
This project implements a comprehensive solution for the Kaggle Facial Keypoints Detection competition. It uses deep learning to detect 15 facial keypoints (30 coordinates) from 96x96 grayscale face images.

### Key Features
- **Multiple CNN Architectures**: BasicCNN, DeepCNN, ResNet-based, and EfficientNet-based models
- **Data Augmentation**: Robust preprocessing with rotation, scaling, brightness adjustments, and noise
- **Experiment Tracking**: ClearML integration for monitoring training progress and metrics
- **Web Application**: Interactive Streamlit app for real-time keypoint detection
- **Comprehensive Testing**: Unit tests for models and data processing pipelines
- **Bilingual Support**: Both Japanese and English interfaces

### Project Structure
```
FaceDetect/
├── .github/workflows/     # GitHub Actions configuration
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   │   ├── dataset.py    # Dataset class with augmentation support
│   │   └── preprocessing.py # Data preprocessing utilities
│   ├── models/           # Model architectures
│   │   └── cnn_model.py  # CNN models (Basic, Deep, ResNet, EfficientNet)
│   ├── training/         # Training pipeline
│   │   ├── trainer.py    # Training class with ClearML integration
│   │   └── train.py      # Main training script
│   └── utils/            # Utility modules
│       ├── visualization.py # Plotting and visualization tools
│       └── inference.py  # Inference utilities
├── webapp/               # Streamlit web application
│   └── app.py           # Main web app with bilingual interface
├── config/               # Configuration files
│   └── clearml.yaml     # ClearML configuration template
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
│   ├── test_models.py   # Model architecture tests
│   └── test_data.py     # Data processing tests
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

### Data Preprocessing Details

#### Normalization
- Images normalized to [0, 1] range with ImageNet statistics (mean=0.485, std=0.229)
- Keypoints normalized to [0, 1] relative to image dimensions

#### Data Augmentation
- **Geometric**: Horizontal flip (50%), rotation (±15°), shift-scale-rotate
- **Photometric**: Random brightness/contrast, Gaussian noise, Gaussian blur
- **Keypoint-aware**: All augmentations preserve keypoint relationships

#### Missing Value Handling
- **Drop**: Remove samples with missing keypoints (default)
- **Interpolate**: Use temporal interpolation for missing values
- **Zero**: Fill missing values with zeros

### Model Architectures

#### BasicCNN
- 4 convolutional blocks with batch normalization
- MaxPooling and dropout for regularization
- 2 fully connected layers
- **Parameters**: ~2.3M

#### DeepCNN
- 5 convolutional blocks with residual-like connections
- Adaptive global average pooling
- Enhanced feature extraction
- **Parameters**: ~8.1M

#### ResNet-based
- Pre-trained ResNet backbones (ResNet18/34/50)
- Modified first layer for grayscale input
- Custom head for keypoint regression
- **Parameters**: 11.2M - 23.5M

#### EfficientNet-based
- Pre-trained EfficientNet backbones (B0/B2)
- Efficient architecture with compound scaling
- Modified for grayscale input
- **Parameters**: 4.0M - 7.7M

### Training Configuration

#### Loss Function
- **MSE Loss**: Mean Squared Error for coordinate regression
- **Alternative**: L1 Loss, Smooth L1 Loss available

#### Optimizer
- **Adam**: Default with learning rate 0.001
- **AdamW**: With weight decay for better generalization
- **SGD**: With momentum for stable training

#### Learning Rate Scheduling
- **StepLR**: Decay by factor of 0.1 every 30 epochs
- **CosineAnnealingLR**: Smooth decay to zero
- **ReduceLROnPlateau**: Adaptive reduction based on validation loss

### Setup Instructions

#### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Kalorie560/FaceDetect.git
cd FaceDetect

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

#### 2. ClearML Configuration
```bash
# Copy and edit the ClearML configuration
cp config/clearml.yaml config/clearml_local.yaml

# Edit config/clearml_local.yaml with your credentials:
# - Replace YOUR_ACCESS_KEY_HERE with your ClearML access key
# - Replace YOUR_SECRET_KEY_HERE with your ClearML secret key
# - Replace YOUR_WORKSPACE_NAME with your workspace name
```

#### 3. Data Preparation

##### Dataset File Placement
Place the competition dataset files in the project root directory:
```
FaceDetect/
├── training.csv        # Training data with images and keypoints
├── test.csv           # Test data with images only
├── IdLookupTable.csv  # Maps image IDs to keypoint labels
└── ... (other project files)
```

##### Download and Setup
```bash
# Download Kaggle data
kaggle competitions download -c facial-keypoints-detection

# Extract data to project root
unzip facial-keypoints-detection.zip -d .

# Verify files are in the correct location
ls -la *.csv
# Should show: training.csv, test.csv, IdLookupTable.csv

# The training.csv file should contain:
# - 'Image' column with space-separated pixel values
# - 30 keypoint coordinate columns
```

### Training the Model

#### Basic Training
```bash
python src/training/train.py \
    --data_path ./training.csv \
    --model_type resnet18 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

#### Advanced Training with ClearML
```bash
python src/training/train.py \
    --data_path ./training.csv \
    --model_type efficientnet_b0 \
    --epochs 150 \
    --batch_size 64 \
    --clearml_config config/clearml_local.yaml \
    --project_name "facial_keypoints_detection" \
    --experiment_name "efficientnet_b0_experiment"
```

### Running the Web Application

```bash
# Start the Streamlit app
streamlit run webapp/app.py

# Open browser to http://localhost:8501
# Upload an image and detect facial keypoints in real-time
```

### Web Application Features

#### Input Requirements
- **Supported Formats**: JPEG, PNG
- **Recommended Size**: 96x96 pixels or larger
- **Optimal Input**: Front-facing images with face centered
- **Single Face**: Works best with one face per image

#### Real-time Detection
- Upload image through web interface
- Automatic face detection and cropping (optional)
- Real-time keypoint prediction and visualization
- Confidence score display
- Coordinate values table
- Results download in JSON format

### Results and Evaluation

#### Performance Metrics
- **MSE**: Mean Squared Error for coordinate accuracy
- **MAE**: Mean Absolute Error for robust evaluation
- **Per-keypoint Analysis**: Individual keypoint accuracy assessment

#### Validation Strategy
- **Train/Val/Test Split**: 70%/20%/10% default
- **Stratified Sampling**: Ensures balanced data distribution
- **Cross-validation**: K-fold validation for robust evaluation

#### Expected Performance
- **BasicCNN**: MSE ~0.005-0.01 on validation set
- **ResNet18**: MSE ~0.003-0.007 on validation set
- **EfficientNet-B0**: MSE ~0.002-0.005 on validation set

### Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_models.py
python -m pytest tests/test_data.py

# Run with coverage
pip install pytest-cov
python -m pytest --cov=src tests/
```

### Additional Considerations

#### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for training
- **Memory**: 8GB+ RAM, 4GB+ GPU memory
- **Storage**: 2GB+ for models and data

#### Performance Optimization
- **Mixed Precision**: Use `--mixed_precision` flag for faster training
- **Data Loading**: Adjust `--num_workers` based on CPU cores
- **Batch Size**: Increase based on available GPU memory

#### Security Best Practices
- Never commit API keys or credentials to the repository
- Use environment variables or secure config files
- Implement proper input validation in the web application

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

### License
This project is open source and available under the MIT License.

---

## Japanese

### 概要
このプロジェクトは、Kaggle顔特徴点検出コンペティションの包括的なソリューションを実装しています。深層学習を使用して、96x96のグレースケール顔画像から15個の顔特徴点（30座標）を検出します。

### 主要機能
- **複数のCNNアーキテクチャ**: BasicCNN、DeepCNN、ResNetベース、EfficientNetベースモデル
- **データ拡張**: 回転、スケーリング、明度調整、ノイズを含む堅牢な前処理
- **実験追跡**: 学習進行とメトリクスの監視のためのClearML統合
- **Webアプリケーション**: リアルタイム特徴点検出のためのインタラクティブなStreamlitアプリ
- **包括的テスト**: モデルとデータ処理パイプラインのユニットテスト
- **バイリンガルサポート**: 日本語と英語の両方のインターフェース

### データ前処理の詳細

#### 正規化手法
- 画像を[0, 1]範囲にImageNet統計（平均=0.485、標準偏差=0.229）で正規化
- 特徴点を画像サイズに対して[0, 1]で正規化

#### データ拡張の種類とパラメータ
- **幾何学的変換**: 水平フリップ（50%）、回転（±15°）、シフト・スケール・回転
- **フォトメトリック**: ランダム明度/コントラスト、ガウシアンノイズ、ガウシアンブラー
- **特徴点対応**: すべての拡張で特徴点の関係を保持

#### 欠損値処理の戦略
- **削除**: 欠損特徴点のあるサンプルを削除（デフォルト）
- **補間**: 欠損値に対して時系列補間を使用
- **ゼロ補完**: 欠損値をゼロで埋める

### モデルアーキテクチャの詳細説明

#### BasicCNN ネットワーク構造
```
Input (1, 96, 96)
├── Conv2d(1→32, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2d(32→64, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2d(64→128, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Conv2d(128→256, 3x3) + BatchNorm + ReLU + MaxPool(2x2)
├── Flatten → Linear(9216→1024) + ReLU + Dropout(0.5)
├── Linear(1024→512) + ReLU + Dropout(0.5)
└── Linear(512→30) [出力層]
```

### 学習設定

#### 損失関数
- **MSE Loss**: 座標回帰のための平均二乗誤差
- **代替案**: L1 Loss、Smooth L1 Lossも利用可能

#### オプティマイザとハイパーパラメータ
- **Adam**: 学習率0.001（デフォルト）
- **重み減衰**: 汎化性能向上のためのAdamW
- **運動量**: 安定した学習のためのSGD

#### 学習率スケジューラ
- **StepLR**: 30エポックごとに0.1倍に減衰
- **CosineAnnealingLR**: ゼロまでの滑らかな減衰
- **ReduceLROnPlateau**: 検証損失に基づく適応的減少

### 環境構築手順

#### 1. 環境セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/Kalorie560/FaceDetect.git
cd FaceDetect

# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate

# 依存関係のインストール
pip install -r requirements.txt
```

#### 2. ClearML設定方法
```bash
# ClearML設定をコピーして編集
cp config/clearml.yaml config/clearml_local.yaml

# config/clearml_local.yamlを認証情報で編集:
# - YOUR_ACCESS_KEY_HEREをClearMLアクセスキーに置換
# - YOUR_SECRET_KEY_HEREをClearMLシークレットキーに置換
# - YOUR_WORKSPACE_NAMEをワークスペース名に置換
```

#### 3. データセットの準備

##### データセットファイルの配置場所
コンペティションのデータセットファイルをプロジェクトのルートディレクトリに配置してください:
```
FaceDetect/
├── training.csv        # 訓練データ（画像と特徴点）
├── test.csv           # テストデータ（画像のみ）
├── IdLookupTable.csv  # 画像IDと特徴点ラベルのマッピング
└── ... (その他のプロジェクトファイル)
```

##### ダウンロードとセットアップ
```bash
# Kaggleデータのダウンロード
kaggle competitions download -c facial-keypoints-detection

# プロジェクトルートにデータを展開
unzip facial-keypoints-detection.zip -d .

# ファイルが正しい場所にあることを確認
ls -la *.csv
# 結果: training.csv, test.csv, IdLookupTable.csv が表示されるはず

# training.csvには以下が含まれています:
# - 'Image'列: スペース区切りの画素値
# - 30個の特徴点座標列
```

### モデル学習の実行方法

#### 基本的な学習
```bash
python src/training/train.py \
    --data_path ./training.csv \
    --model_type resnet18 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Webアプリの起動方法

```bash
# Streamlitアプリの起動
streamlit run webapp/app.py

# ブラウザでhttp://localhost:8501を開く
# 画像をアップロードしてリアルタイムで顔特徴点を検出
```

### 結果と評価

#### 検証データでの性能
- **BasicCNN**: 検証セットでMSE約0.005-0.01
- **ResNet18**: 検証セットでMSE約0.003-0.007
- **EfficientNet-B0**: 検証セットでMSE約0.002-0.005

#### 予測結果の可視化例
- Webアプリケーションでリアルタイム可視化
- 特徴点の座標値表示
- 信頼度スコアの表示
- JSON形式での結果ダウンロード

### テスト実行

```bash
# すべてのテストを実行
python -m pytest tests/

# 特定のテストモジュールを実行
python -m pytest tests/test_models.py
python -m pytest tests/test_data.py
```

### 追加の考慮事項

#### ハードウェア要件
- **GPU**: 学習にはCUDA対応GPUを推奨
- **メモリ**: 8GB以上のRAM、4GB以上のGPUメモリ
- **ストレージ**: モデルとデータ用に2GB以上

#### セキュリティのベストプラクティス
- APIキーや認証情報をリポジトリにコミットしない
- 環境変数やセキュアな設定ファイルを使用
- Webアプリケーションで適切な入力検証を実装

### ライセンス
このプロジェクトはオープンソースで、MITライセンスの下で利用可能です。