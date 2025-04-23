import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

def view_pickle(filepath, num_samples = 3):
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n{'='*50}")
        print(f"File: {Path(filepath).name}")
        print(f"Total samples: {len(data)}")
        print(f"Data type: {type(data).__name__}")
        
        if isinstance(data, (list, tuple)):
            print("\nSample entries:")
            for i in range(min(num_samples, len(data))):
                print(f"\nSample {i}:")
                if isinstance(data[i], dict):
                    print(pd.DataFrame.from_dict(data[i], orient = 'index'))
                else:
                    print(data[i])
        
        elif isinstance(data, dict):
            print("\nDictionary contents:")
            print(f"Keys: {list(data.keys())}")
            print("\nSample values:")
            for i, (k, v) in enumerate(data.items()):
                if i >= num_samples:
                    break
                print(f"\nKey: {k}")
                print(pd.DataFrame.from_dict(v, orient = 'index') if isinstance(v, dict) else v)
        
        else:
            print("\nData preview:")
            print(data[:num_samples] if hasattr(data, '__getitem__') else data)

    except Exception as e:
        print(f"\nError loading {filepath}: {str(e)}")

def analyze_hypernym_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\n{'='*50}")
        print(f"Analysis for: {Path(filepath).name}")
        print(f"Total samples: {len(data)}")
        
        stats = {
            'avg_hypernyms_per_term': 0,
            'unique_hyponyms': set(),
            'unique_hypernyms': set(),
            'hypernym_distribution': defaultdict(int)
        }
        
        for entry in data:
            hyponym = entry['children']
            hypernyms = entry['parents']
            
            stats['unique_hyponyms'].add(hyponym)
            stats['avg_hypernyms_per_term'] += len(hypernyms)
            
            for hyper in hypernyms:
                stats['unique_hypernyms'].add(hyper)
                stats['hypernym_distribution'][hyper] += 1
        
        stats['avg_hypernyms_per_term'] /= len(data)
        stats['unique_hyponyms'] = len(stats['unique_hyponyms'])
        stats['unique_hypernyms'] = len(stats['unique_hypernyms'])
        
        print(f"Unique hyponyms: {stats['unique_hyponyms']}")
        print(f"Unique hypernyms: {stats['unique_hypernyms']}")
        print(f"Avg hypernyms per term: {stats['avg_hypernyms_per_term']:.2f}")
        
        top_hypernyms = sorted(stats['hypernym_distribution'].items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 most common hypernyms:")
        for hyper, count in top_hypernyms:
            print(f"{hyper}: {count} occurrences")
            
        return stats
        
    except Exception as e:
        print(f"Error analyzing {filepath}: {str(e)}")
        return None

if __name__ == "__main__":
    files_to_view = [
        "Data/HypernymDiscovery/1A.english.pickle",
        "Data/HypernymDiscovery/1A.english_train.pickle",
        "Data/HypernymDiscovery/1B.italian.pickle",
    ]

    all_stats = {}
    for file in files_to_view:
        stats = analyze_hypernym_data(file)
        if stats:
            all_stats[Path(file).name] = stats