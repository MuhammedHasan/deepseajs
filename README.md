# DeepSeaJS

Run DeepSea genomics model with per base pair nucleotide features in your browser. No server required. Your data never leaves your computer. No installation or configuration required.

See demo at [https://deepseajs.vercel.app/](https://deepseajs.vercel.app/)

This is for prototype purposes only and not intended for production use cases. 

# Model training

Install latest dependencies `conda env create -f workflow/environment.yml`

Generate train, benchmark and generate onnx file for the model

```bash
snakemake -j 16 
```

# Run

Store generated onnx model under the src/public directory. Then run the react app:

```bash
npm run dev
```

or build and bundle to obtain static files:

```bash
npm run build
```

# Credits

Original DeepSea Paper: [doi.org/10.1038/nmeth.3547](https://doi.org/10.1038/nmeth.3547)