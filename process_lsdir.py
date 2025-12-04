import os
import pandas as pd
from PIL import Image
import io
import multiprocessing as mp

def extract_image_data(image_info):
    """
    Extract byte data from image information.
    
    Args:
        image_info (dict): Dictionary containing image data, e.g. {'bytes': b'...'}

    Returns:
        bytes: Image byte data, or None if extraction fails
    """
    if isinstance(image_info, dict):
        return image_info.get('bytes')
    elif isinstance(image_info, bytes):
        return image_info
    else:
        return None

def process_parquet_file(parquet_file, output_dir):
    print(f"Processing file: {parquet_file}")
    try:
        df = pd.read_parquet(parquet_file)
    except Exception as e:
        print(f"Unable to read Parquet file {parquet_file}: {e}")
        return
    
    for idx, row in df.iterrows():
        image_info = row['image']  # The 'image' column of current row is a dictionary
        image_data = extract_image_data(image_info)
        
        if image_data is None:
            print(f"Image data missing or cannot be parsed: {row.get('path', 'unknown path')}")
            continue
        
        image_path = row.get('path')
        if not image_path:
            print(f"Image save path not found: {idx}")
            continue
        
        save_path = os.path.join(output_dir, image_path)
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            image.save(save_path)
            
            # Optional: verify if the saved image is valid
            with Image.open(save_path) as img_verify:
                img_verify.verify()
        except Exception as e:
            print(f"Unable to save or verify image {save_path}: {e}")

def convert_all_parquet_to_images(parquet_dir, output_dir, num_workers=4):
    """
    Extract and save image data from all Parquet files in the specified directory.
    
    Args:
        parquet_dir (str): Directory path containing Parquet files.
        output_dir (str): Target directory path to save extracted images.
        num_workers (int): Number of parallel processing processes.
    """
    parquet_files = [os.path.join(parquet_dir, f) for f in os.listdir(parquet_dir) if f.endswith('.parquet')]
    
    if not parquet_files:
        print(f"No Parquet files found in directory {parquet_dir}.")
        return
    
    # Use multiprocessing to accelerate processing
    pool = mp.Pool(processes=num_workers)
    results = [pool.apply_async(process_parquet_file, args=(pf, output_dir)) for pf in parquet_files]
    
    # Wait for all tasks to complete
    for r in results:
        r.get()
    
    pool.close()
    pool.join()
    print("All files processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Parquet files to images.")
    parser.add_argument("--parquet_dir", type=str, required=True, help="Directory containing Parquet files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save extracted images.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker processes.")
    args = parser.parse_args()
    number_of_workers = args.num_workers  # Set according to your CPU core count and I/O bandwidth
    parquet_directory = args.parquet_dir
    images_output_directory = args.output_dir

    convert_all_parquet_to_images(parquet_directory, images_output_directory, num_workers=number_of_workers)
