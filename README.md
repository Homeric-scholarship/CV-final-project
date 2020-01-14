# CV-final-project
## Usage
### Run Example
    python3 main.py --input-left <Lpath> --input-right <Rpath> --output <Outputpath>
 - `<Lpath>` (e.g. `.data/Synthetic/TL0.png`)
 - `<Rpath>` (e.g. `.data/Synthetic/TR0.png`)
 - `<Outputpath>` (e.g. `.output/Synthetic/SD0.pfm`)

### Visualize
    python3 run_visualize.py

### Test
    python3 run_main.py <Scene> 
 - `<Scene>` (e.g. `Synthetic`)

### Evaluation
    python3 cal_err.py 
Calculate average error on synthetic scene.

