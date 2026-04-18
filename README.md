# whisper-subset-selection

## 📂 Project Structure
.
├── audioToVec.py # Extract Whisper embeddings
├── selectKsamples.py # K-center subset selection
├── selectLongestAudio.py # Longest-audio baseline
├── train.py # Train Whisper model
├── lexCoverage.py # Lexical diversity analysis
├── plot.py # Plots (data + results)
├── common_voice_subset/ # Preprocessed datasets
└── data4/ # Embeddings / indices


## Examples
### Extract Embeddings 
```
python audioToVec.py \
  --language-name Danish \
  --language-code da
```

### Find Longest K samples 
```
python selectLongestAudio.py \
  --language-name Hebrew \
  --language-code he \
  --top-k-values 500 1000
```

### Run Subset Selection
```
python selectKsamples.py \
  --language danish \
  --method cosine \
  --k-values 500 1000 2000 5000
```
### Train Whisper
```
python train_whisper.py \
  --language-name Hebrew \
  --language-code he \
  --subset-indices-path kcenter/selected_indices_cosinek5000.txt
```
### Lexical coverage analysis
```
python lexCoverage.py \
  --language-code he \
  --indices-path kcenter/selected_indices_cosinek2000.txt
```
