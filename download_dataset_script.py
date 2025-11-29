"""
Auto-download fake news dataset
Run this script to automatically download and prepare the dataset
"""

import pandas as pd
import urllib.request
import os

print("Downloading fake news dataset...")

try:
    # Try downloading from GitHub repository
    url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/news.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download the file
    urllib.request.urlretrieve(url, 'news.csv')
    print("✓ Downloaded news.csv successfully!")
    
    # Verify the file
    df = pd.read_csv('news.csv')
    print(f"\nDataset loaded: {len(df)} articles")
    print(f"Columns: {list(df.columns)}")
    
    # Check label distribution
    if 'label' in df.columns:
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
    
    print("\n✓ Dataset ready to use!")
    print("You can now run your training script.")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\nCreating sample dataset instead...")
    
    # Create sample dataset as backup
    real_news = [
        "Scientists at MIT developed new carbon capture technology.",
        "Federal Reserve announces interest rate decision.",
        "Study shows exercise reduces heart disease risk.",
        "UN climate summit reaches emission agreements.",
        "Solar panel efficiency improves by 15 percent.",
        "WHO reports decline in malaria cases globally.",
        "NASA telescope captures images of distant galaxies.",
        "Economic growth forecast at 2.5 percent.",
        "Vaccination rates increase in developing nations.",
        "College graduation rates rise by 5 percent.",
    ] * 25  # Repeat to get 250 samples
    
    fake_news = [
        "SHOCKING: Coffee makes you immortal, doctors hide this!",
        "Celebrity reveals vaccine truth they don't want you to know!",
        "BREAKING: Aliens built pyramids, government documents prove it!",
        "Miracle cure eliminates all diseases overnight!",
        "EXPOSED: Smartphone reads your mind!",
        "Man loses 100 pounds in one week eating ice cream!",
        "Scientists confirm Earth is flat, NASA lied!",
        "Billionaire's secret to getting rich overnight!",
        "Tap water contains mind control chemicals!",
        "Doctors discover sleeping secret, pillow industry panics!",
    ] * 25  # Repeat to get 250 samples
    
    # Create DataFrame
    data = []
    for text in real_news:
        data.append({'text': text, 'label': 0})
    for text in fake_news:
        data.append({'text': text, 'label': 1})
    
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save
    df.to_csv('news.csv', index=False)
    print(f"✓ Created sample dataset: {len(df)} articles")
    print("✓ Saved as news.csv")

print("\n" + "="*50)
print("Next steps:")
print("1. Run your training script")
print("2. The script should now find news.csv")
print("="*50)
