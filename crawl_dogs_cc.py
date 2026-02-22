"""
Dog Image Crawler for Copyright-Free Images
Downloads images from Wikimedia Commons and other copyright-free sources
Creates a source file for attribution and verification
"""

import os
import requests
import json
import time
import csv
from urllib.parse import quote, unquote
from pathlib import Path
import hashlib

class DogImageCrawler:
    def __init__(self, output_dir="crawled_images", pexels_api_key=None, pixabay_api_key=None, unsplash_access_key=None):
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DogBreedViT/1.0 (Educational Project; Contact: your-email@example.com)'
        })
        
        # API Keys (optional - will use if provided)
        self.pexels_api_key = pexels_api_key
        self.pixabay_api_key = pixabay_api_key
        self.unsplash_access_key = unsplash_access_key
        
        # Target breeds
        self.breeds = [
            # "German_Shepherd",
            # "Rottweiler",
            # "Boxer",
            # "Bernese_Mountain_Dog",
            "Great_Dane",
            # "Doberman"
        ]
        
        # Track downloaded URLs to avoid duplicates
        self.downloaded_urls = {}
        
        # Wikimedia Commons search terms
        self.search_terms = {
            # "German_Shepherd": ["German Shepherd Dog", "Deutscher Schäferhund"],
            # "Rottweiler": ["Rottweiler", "Rottweiler dog"],
            # "Boxer": ["Boxer dog", "Boxer breed"],
            # "Bernese_Mountain_Dog": ["Bernese Mountain Dog", "Berner Sennenhund"],
            "Great_Dane": ["Great Dane", "Deutsche Dogge"],
            # "Doberman": ["Doberman Pinscher", "Dobermann"]
        }
        
    def create_directories(self):
        """Create output directories for each breed"""
        for breed in self.breeds:
            breed_dir = os.path.join(self.output_dir, breed)
            os.makedirs(breed_dir, exist_ok=True)
            print(f"Created directory: {breed_dir}")
    
    def load_existing_sources(self, breed):
        """Load previously downloaded image URLs from sources file"""
        sources_file = os.path.join(self.output_dir, breed, f"{breed}_sources.csv")
        urls = set()
        
        if os.path.exists(sources_file):
            try:
                with open(sources_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Handle both old format (without Source) and new format (with Source)
                        urls.add(row.get('Source_URL', ''))
                print(f"Found {len(urls)} previously downloaded images for {breed}")
            except Exception as e:
                print(f"Error loading existing sources: {e}")
        
        return urls
    
    def get_image_hash(self, filepath):
        """Calculate MD5 hash of an image file for duplicate detection"""
        try:
            hash_md5 = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error calculating hash: {e}")
            return None
    
    def search_wikimedia_commons(self, search_term, limit=100, offset=0):
        """
        Search Wikimedia Commons for images with pagination support
        API Documentation: https://commons.wikimedia.org/w/api.php
        """
        url = "https://commons.wikimedia.org/w/api.php"
        
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'search',
            'gsrnamespace': '6',  # File namespace
            'gsrsearch': f'filetype:bitmap {search_term}',
            'gsrlimit': min(limit, 50),  # API max is 50
            'gsroffset': offset,
            'prop': 'imageinfo',
            'iiprop': 'url|size|mime|extmetadata',
            'iiurlwidth': '1024'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            images = []
            if 'query' in data and 'pages' in data['query']:
                for page_id, page_data in data['query']['pages'].items():
                    if 'imageinfo' in page_data:
                        image_info = page_data['imageinfo'][0]
                        
                        # Check for copyright-free licenses
                        license_info = self.extract_license_info(image_info.get('extmetadata', {}))
                        
                        if self.is_license_acceptable(license_info):
                            images.append({
                                'title': page_data.get('title', ''),
                                'url': image_info.get('url', ''),
                                'thumb_url': image_info.get('thumburl', image_info.get('url', '')),
                                'width': image_info.get('width', 0),
                                'height': image_info.get('height', 0),
                                'size': image_info.get('size', 0),
                                'mime': image_info.get('mime', ''),
                                'page_url': f"https://commons.wikimedia.org/wiki/{quote(page_data.get('title', ''))}",
                                'license': license_info['license'],
                                'author': license_info['author'],
                                'attribution': license_info['attribution'],
                                'source': 'Wikimedia Commons'
                            })
            
            return images
        
        except Exception as e:
            print(f"Error searching Wikimedia Commons: {e}")
            return []
    
    def search_rawpixel(self, search_term, limit=100):
        """
        Search rawpixel.com for free public domain and CC0 images
        No API key required - web scraping fallback
        rawpixel has high-quality curated free images
        """
        try:
            # rawpixel's public domain collection
            url = f"https://www.rawpixel.com/search/{quote(search_term)}?filter=public-domain&page=1&sort=curated"
            
            # This is a simplified approach - in production, you'd want proper HTML parsing
            # For now, we'll return empty and rely on other sources
            # To implement: would need BeautifulSoup to parse their website
            
            return []
        
        except Exception as e:
            print(f"Error searching rawpixel: {e}")
            return []
    
    def search_pexels(self, search_term, limit=50):
        """
        Search Pexels for free stock photos
        API: https://www.pexels.com/api/
        All Pexels photos are free to use (Pexels License)
        """
        if not self.pexels_api_key:
            return []
        
        url = "https://api.pexels.com/v1/search"
        headers = {
            'Authorization': self.pexels_api_key
        }
        
        params = {
            'query': search_term,
            'per_page': min(limit, 80),  # API max is 80
            'page': 1
        }
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for photo in data.get('photos', []):
                images.append({
                    'title': f"Pexels Photo {photo['id']}",
                    'url': photo['src']['original'],
                    'thumb_url': photo['src']['large'],  # 1280px wide
                    'width': photo['width'],
                    'height': photo['height'],
                    'size': 0,  # Unknown for Pexels
                    'mime': 'image/jpeg',
                    'page_url': photo['url'],
                    'license': 'Pexels License (Free to use)',
                    'author': photo['photographer'],
                    'attribution': f"Photo by {photo['photographer']} on Pexels",
                    'source': 'Pexels'
                })
            
            return images
        
        except Exception as e:
            print(f"Error searching Pexels: {e}")
            return []
    
    def search_pixabay(self, search_term, limit=50):
        """
        Search Pixabay for free images
        API: https://pixabay.com/api/docs/
        All Pixabay images are free (Pixabay License)
        """
        if not self.pixabay_api_key:
            return []
        
        url = "https://pixabay.com/api/"
        
        params = {
            'key': self.pixabay_api_key,
            'q': search_term,
            'image_type': 'photo',
            'per_page': min(limit, 200),  # API max is 200
            'safesearch': 'true'
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for hit in data.get('hits', []):
                images.append({
                    'title': f"Pixabay Image {hit['id']}",
                    'url': hit['largeImageURL'],
                    'thumb_url': hit['webformatURL'],  # 640px wide
                    'width': hit['imageWidth'],
                    'height': hit['imageHeight'],
                    'size': hit.get('imageSize', 0),
                    'mime': 'image/jpeg',
                    'page_url': hit['pageURL'],
                    'license': 'Pixabay License (Free to use)',
                    'author': hit['user'],
                    'attribution': f"Image by {hit['user']} from Pixabay",
                    'source': 'Pixabay'
                })
            
            return images
        
        except Exception as e:
            print(f"Error searching Pixabay: {e}")
            return []
    
    def search_unsplash(self, search_term, limit=30):
        """
        Search Unsplash for high-quality free photos
        API: https://unsplash.com/developers
        All Unsplash photos are free to use (Unsplash License)
        """
        if not self.unsplash_access_key:
            return []
        
        url = "https://api.unsplash.com/search/photos"
        headers = {
            'Authorization': f'Client-ID {self.unsplash_access_key}'
        }
        
        params = {
            'query': search_term,
            'per_page': min(limit, 30),  # API max is 30
            'page': 1,
            'orientation': 'landscape'  # Better for dog photos
        }
        
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            images = []
            for photo in data.get('results', []):
                images.append({
                    'title': f"Unsplash Photo {photo['id']}",
                    'url': photo['urls']['full'],
                    'thumb_url': photo['urls']['regular'],  # ~1080px wide
                    'width': photo['width'],
                    'height': photo['height'],
                    'size': 0,  # Unknown for Unsplash
                    'mime': 'image/jpeg',
                    'page_url': photo['links']['html'],
                    'license': 'Unsplash License (Free to use)',
                    'author': photo['user']['name'],
                    'attribution': f"Photo by {photo['user']['name']} on Unsplash",
                    'source': 'Unsplash'
                })
            
            return images
        
        except Exception as e:
            print(f"Error searching Unsplash: {e}")
            return []
    
    def search_all_sources(self, search_term, limit=150):
        """
        Search multiple sources and combine results
        Uses pagination and multiple search variations for diversity
        """
        all_images = []
        
        # 1. Search Wikimedia Commons with pagination
        print(f"  Searching Wikimedia Commons (page 1)...")
        images = self.search_wikimedia_commons(search_term, limit=50, offset=0)
        all_images.extend(images)
        
        if len(all_images) < limit:
            print(f"  Searching Wikimedia Commons (page 2)...")
            time.sleep(1)
            images = self.search_wikimedia_commons(search_term, limit=50, offset=50)
            all_images.extend(images)
        
        if len(all_images) < limit:
            print(f"  Searching Wikimedia Commons (page 3)...")
            time.sleep(1)
            images = self.search_wikimedia_commons(search_term, limit=50, offset=100)
            all_images.extend(images)
        
        # 2. Search Pexels if API key is available
        if self.pexels_api_key and len(all_images) < limit:
            print(f"  Searching Pexels...")
            time.sleep(1)
            images = self.search_pexels(search_term, limit=80)
            all_images.extend(images)
            print(f"    Found {len(images)} images from Pexels")
        
        # 3. Search Pixabay if API key is available
        if self.pixabay_api_key and len(all_images) < limit:
            print(f"  Searching Pixabay...")
            time.sleep(1)
            images = self.search_pixabay(search_term, limit=100)
            all_images.extend(images)
            print(f"    Found {len(images)} images from Pixabay")
        
        # 4. Search Unsplash if API key is available
        if self.unsplash_access_key and len(all_images) < limit:
            print(f"  Searching Unsplash...")
            time.sleep(1)
            images = self.search_unsplash(search_term, limit=30)
            all_images.extend(images)
            print(f"    Found {len(images)} images from Unsplash")
        
        # 5. Try with MORE specific search queries for better matching
        alternate_queries = [
            f"{search_term} dog",
            f"{search_term} breed",
            f"{search_term} full body",
            f"{search_term} side view",
            f"{search_term} adult",
            f"{search_term} close up"
        ]
        
        for alt_query in alternate_queries:
            if len(all_images) >= limit * 2:  # Get extra for variety
                break
            print(f"  Searching with variation: {alt_query}")
            time.sleep(1)
            
            # Search Wikimedia with variations
            images = self.search_wikimedia_commons(alt_query, limit=30, offset=0)
            all_images.extend(images)
            
            # Also try Pexels/Pixabay/Unsplash with variations
            if self.pexels_api_key:
                time.sleep(1)
                images = self.search_pexels(alt_query, limit=20)
                all_images.extend(images)
            
            if self.pixabay_api_key:
                time.sleep(1)
                images = self.search_pixabay(alt_query, limit=30)
                all_images.extend(images)
            
            if self.unsplash_access_key:
                time.sleep(1)
                images = self.search_unsplash(alt_query, limit=15)
                all_images.extend(images)
        
        # Remove duplicates based on URL
        unique_images = []
        seen_urls = set()
        for img in all_images:
            if img['url'] not in seen_urls:
                seen_urls.add(img['url'])
                unique_images.append(img)
        
        return unique_images
    
    def extract_license_info(self, extmetadata):
        """Extract license information from image metadata"""
        license_info = {
            'license': 'Unknown',
            'author': 'Unknown',
            'attribution': 'Unknown'
        }
        
        if 'LicenseShortName' in extmetadata:
            license_info['license'] = extmetadata['LicenseShortName'].get('value', 'Unknown')
        elif 'License' in extmetadata:
            license_info['license'] = extmetadata['License'].get('value', 'Unknown')
        
        if 'Artist' in extmetadata:
            license_info['author'] = extmetadata['Artist'].get('value', 'Unknown')
        elif 'Credit' in extmetadata:
            license_info['author'] = extmetadata['Credit'].get('value', 'Unknown')
        
        if 'AttributionRequired' in extmetadata:
            license_info['attribution'] = extmetadata['AttributionRequired'].get('value', 'Unknown')
        
        return license_info
    
    def is_license_acceptable(self, license_info):
        """
        Check if the license allows reuse
        Acceptable licenses: Public Domain, CC0, CC BY, CC BY-SA
        """
        license_text = license_info['license'].upper()
        
        acceptable_licenses = [
            'PUBLIC DOMAIN',
            'CC0',
            'CC BY',
            'CC-BY',
            'CC BY-SA',
            'CC-BY-SA',
            'PD',
            'FREE ART LICENSE'
        ]
        
        return any(lic in license_text for lic in acceptable_licenses)
    
    def download_image(self, image_data, breed, image_number, existing_hashes=None):
        """Download a single image with duplicate detection"""
        if existing_hashes is None:
            existing_hashes = set()
        
        try:
            # Use thumb URL for reasonable file sizes
            url = image_data['thumb_url']
            
            response = self.session.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Generate filename
            extension = image_data['mime'].split('/')[-1]
            if extension not in ['jpg', 'jpeg', 'png', 'webp']:
                extension = 'jpg'
            
            filename = f"{breed}_{image_number:03d}.{extension}"
            filepath = os.path.join(self.output_dir, breed, filename)
            
            # Download image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # Verify file size
            file_size = os.path.getsize(filepath)
            if file_size < 5000:  # Less than 5KB might be an error
                os.remove(filepath)
                return None
            
            # Check for duplicate content
            image_hash = self.get_image_hash(filepath)
            if image_hash and image_hash in existing_hashes:
                print(f" [duplicate content detected]", end='')
                os.remove(filepath)
                return None
            
            if image_hash:
                existing_hashes.add(image_hash)
            
            return {
                'filename': filename,
                'filepath': filepath,
                'file_size': file_size,
                'hash': image_hash,
                **image_data
            }
        
        except Exception as e:
            print(f"Error downloading image: {e}")
            return None
    
    def save_sources_file(self, breed, downloaded_images, starting_number=1):
        """Save source information to CSV file (append mode for new images)"""
        sources_file = os.path.join(self.output_dir, breed, f"{breed}_sources.csv")
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(sources_file)
        
        with open(sources_file, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header only if creating new file
            if not file_exists:
                writer.writerow([
                    'Image_Number',
                    'Filename',
                    'Source',
                    'Source_URL',
                    'Page_URL',
                    'License',
                    'Author',
                    'File_Size_Bytes',
                    'Dimensions',
                    'Downloaded_At'
                ])
            
            # Data rows (with correct numbering)
            for idx, img in enumerate(downloaded_images, starting_number):
                writer.writerow([
                    idx,
                    img['filename'],
                    img.get('source', 'Wikimedia Commons'),
                    img['url'],
                    img['page_url'],
                    img['license'],
                    img['author'],
                    img['file_size'],
                    f"{img['width']}x{img['height']}",
                    time.strftime('%Y-%m-%d %H:%M:%S')
                ])
        
        print(f"Sources {'appended to' if file_exists else 'saved to'}: {sources_file}")
        
        # Also update README with attribution information
        readme_file = os.path.join(self.output_dir, breed, f"{breed}_README.txt")
        readme_exists = os.path.exists(readme_file)
        
        with open(readme_file, 'a' if readme_exists else 'w', encoding='utf-8') as f:
            if not readme_exists:
                # Write header for new file
                f.write(f"Image Sources for {breed.replace('_', ' ')}\n")
                f.write("=" * 60 + "\n\n")
                f.write("All images are from Wikimedia Commons and are licensed under\n")
                f.write("Creative Commons or Public Domain licenses.\n\n")
                f.write("Attribution:\n")
                f.write("-" * 60 + "\n\n")
            else:
                # Add separator for new batch
                f.write(f"\n{'='*60}\n")
                f.write(f"Additional images downloaded: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
            
            for idx, img in enumerate(downloaded_images, starting_number):
                f.write(f"{idx}. {img['filename']}\n")
                f.write(f"   Source Platform: {img.get('source', 'Wikimedia Commons')}\n")
                f.write(f"   Page: {img['page_url']}\n")
                f.write(f"   License: {img['license']}\n")
                f.write(f"   Author: {img['author']}\n")
                f.write(f"   Direct URL: {img['url']}\n\n")
        
        print(f"README {'appended to' if readme_exists else 'saved to'}: {readme_file}")
    
    def crawl_breed(self, breed, min_images=50, force_continue=False):
        """Crawl images for a specific breed with duplicate detection"""
        print(f"\n{'='*60}")
        print(f"Crawling images for: {breed.replace('_', ' ')}")
        print(f"{'='*60}")
        
        # Count actual image files in directory (this is the real count!)
        breed_dir = os.path.join(self.output_dir, breed)
        actual_image_count = 0
        if os.path.exists(breed_dir):
            actual_image_count = len([f for f in os.listdir(breed_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
        
        # Load existing URLs to skip (for deduplication)
        existing_urls = self.load_existing_sources(breed)
        csv_count = len(existing_urls)
        
        print(f"Current status: {actual_image_count} images in folder, {csv_count} in CSV")
        
        # Use actual folder count, not CSV count, to determine if we need more images
        if actual_image_count >= min_images and not force_continue:
            print(f"Already reached target of {min_images} images (based on folder count)")
            print(f"To download more, increase min_images in main() or set force_continue=True")
            return 0
        
        # Calculate how many more images we need (based on actual folder count)
        needed_images = min_images - actual_image_count
        print(f"Need {needed_images} more images to reach {min_images} (currently have {actual_image_count} in folder)")
        
        # Get the next image number based on what's actually in the folder
        max_existing_number = 0
        if os.path.exists(breed_dir):
            for filename in os.listdir(breed_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    # Extract number from filename like "Great_Dane_042.jpg"
                    try:
                        num = int(filename.split('_')[-1].split('.')[0])
                        max_existing_number = max(max_existing_number, num)
                    except:
                        pass
        
        next_image_number = max_existing_number + 1
        print(f"Next image will be numbered: {next_image_number}")
        
        # Load existing image hashes from directory
        existing_hashes = set()
        if os.path.exists(breed_dir):
            for filename in os.listdir(breed_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    filepath = os.path.join(breed_dir, filename)
                    img_hash = self.get_image_hash(filepath)
                    if img_hash:
                        existing_hashes.add(img_hash)
            print(f"Loaded {len(existing_hashes)} existing image hashes")
        
        downloaded_images = []
        search_terms = self.search_terms.get(breed, [breed.replace('_', ' ')])
        skipped_duplicates = 0
        
        for search_term in search_terms:
            if len(downloaded_images) >= needed_images:
                break
            
            print(f"\nSearching for: {search_term}")
            images = self.search_all_sources(search_term, limit=200)
            print(f"Found {len(images)} unique potential images")
            
            for image_data in images:
                if len(downloaded_images) >= needed_images:
                    break
                
                # Skip if URL already downloaded
                if image_data['url'] in existing_urls or image_data['thumb_url'] in existing_urls:
                    skipped_duplicates += 1
                    continue
                
                current_number = next_image_number + len(downloaded_images)
                print(f"Downloading image {current_number}/{next_image_number + needed_images - 1}...", end=' ')
                
                # Use correct image number (continuing from max existing)
                result = self.download_image(image_data, breed, current_number, existing_hashes)
                
                if result:
                    downloaded_images.append(result)
                    existing_urls.add(result['url'])
                    print("✓")
                else:
                    print("✗ (failed or duplicate)")
                
                # Be respectful to the API
                time.sleep(0.5)
        
        print(f"\n{'='*60}")
        print(f"Downloaded {len(downloaded_images)} NEW images for {breed}")
        print(f"Skipped {skipped_duplicates} duplicate URLs")
        print(f"Total images now: {actual_image_count + len(downloaded_images)}")
        print(f"{'='*60}")
        
        if downloaded_images:
            # Append to existing sources file with correct starting number
            self.save_sources_file(breed, downloaded_images, starting_number=next_image_number)
        
        return len(downloaded_images)
    
    def crawl_all_breeds(self, min_images=50, force_continue=False):
        """Crawl images for all breeds"""
        self.create_directories()
        
        results = {}
        total_images = 0
        
        print(f"\nStarting image crawler for {len(self.breeds)} breeds")
        print(f"Target: {min_images} images per breed")
        if force_continue:
            print(f"Force continue mode: Will try to download even if target reached\n")
        else:
            print(f"Standard mode: Will skip breeds that reached target\n")
        
        for breed in self.breeds:
            count = self.crawl_breed(breed, min_images, force_continue)
            results[breed] = count
            total_images += count
        
        # Print summary
        print("\n" + "="*60)
        print("CRAWLING SUMMARY")
        print("="*60)
        for breed, count in results.items():
            status = "✓" if count >= min_images else "✗"
            print(f"{status} {breed.replace('_', ' ')}: {count} images")
        print(f"\nTotal images downloaded: {total_images}")
        print("="*60)
        
        # Save overall summary
        summary_file = os.path.join(self.output_dir, "crawl_summary.json")
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_images': total_images,
                'min_images_target': min_images,
                'breeds': results
            }, f, indent=2)
        
        print(f"\nSummary saved to: {summary_file}")


def main():
    """Main function to run the crawler"""
    
    # ========================================================================
    # OPTIONAL: Add your FREE API keys here for more image sources!
    # ========================================================================
    # Get Pexels API key (FREE): https://www.pexels.com/api/
    # Get Pixabay API key (FREE): https://pixabay.com/api/docs/
    # Get Unsplash Access Key (FREE): https://unsplash.com/developers
    
    PEXELS_API_KEY = "OJve7Hxx1tIFiHIdlONfvcDDW6s3HDHJUhwtBgFB2ecjoXe4BIWBgOPn"
    PIXABAY_API_KEY = "52933378-6c3f317d9d1b821e1db795f99"
    UNSPLASH_ACCESS_KEY = "SckKU_39nWxfVZ5BvwOX_TO3wCFZG5YkeKoLMufuiWk"
    
    # Create crawler with API keys (leave None if you don't have them)
    crawler = DogImageCrawler(
        output_dir="crawled_images",
        pexels_api_key=PEXELS_API_KEY,
        pixabay_api_key=PIXABAY_API_KEY,
        unsplash_access_key=UNSPLASH_ACCESS_KEY
    )
    
    # Show which sources are active
    sources = ["Wikimedia Commons"]
    if PEXELS_API_KEY:
        sources.append("Pexels")
    if PIXABAY_API_KEY:
        sources.append("Pixabay")
    if UNSPLASH_ACCESS_KEY:
        sources.append("Unsplash")
    
    print("\n" + "="*60)
    print("DOG IMAGE CRAWLER - MULTI-SOURCE EDITION")
    print("="*60)
    print(f"Active sources: {', '.join(sources)}")
    print("\nTIP: For better results, the crawler uses:")
    print("   • Multiple search variations per breed")
    print("   • Content-based duplicate detection")
    print("   • 4 different image sources for variety")
    print("="*60 + "\n")

    # OPTION 1: Download until you reach 100 images per breed (skips if already at 100)
    # crawler.crawl_all_breeds(min_images=100, force_continue=False)
    
    # OPTION 2: Download 150 images per breed (will add 50 more if you have 100)
    # crawler.crawl_all_breeds(min_images=150, force_continue=False)

    # OPTION 3: Download up to 1000 images per breed
    crawler.crawl_all_breeds(min_images=1000, force_continue=False)
    
    print("\n✓ Crawling complete!")
    print(f"Images saved in: {crawler.output_dir}")
    print("Check the *_sources.csv files in each breed folder for attribution details.")


if __name__ == "__main__":
    main()
