import argparse
from main import get_features, TSNEVisualizer, fix_random_seeds

def main():
    parser = argparse.ArgumentParser(description='t-SNE visualization for dataset checking')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for feature extraction')
    parser.add_argument('--num_images', type=int, default=3000, help='Maximum number of images to process')
    args = parser.parse_args()

    fix_random_seeds()

    # Feature extraction
    features, labels, image_paths = get_features(
        args.dataset_path, 
        args.batch_size, 
        args.num_images
    )
    print(f'Features extracted, shape: {features.shape}')

    # Visualization
    visualizer = TSNEVisualizer(features, labels, image_paths)

if __name__ == '__main__':
    main()
