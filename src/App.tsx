import * as ort from 'onnxruntime-web'
import { useEffect, useState } from 'react'
import BarChart, { BarChartProps } from './BarChart'
import tfNames from './tfNames'
import tfs from './tfs'
// @ts-ignore
import { DNALogo } from 'logojs-react';

function oneHotEncoder(dna: string) {
  const encoder: { [key: string]: number[] } = {
    A: [1, 0, 0, 0],
    T: [0, 1, 0, 0],
    C: [0, 0, 1, 0],
    G: [0, 0, 0, 1],
  }
  let data: number[] = [];

  for (let i = 0; i < 4; i++) {
    data = data.concat(dna.split('').map(nc => encoder[nc][i]));
  }

  console.log(data)

  return new ort.Tensor('float32', data, [1, 4, 1000]);
}

function App() {
  const [downloading, setDownloading] = useState(true)
  const [dna, setDna] = useState(tfs[0].sequence)
  const [model, setModel] = useState<ort.InferenceSession | null>(null)
  const [probs, setProbs] = useState<BarChartProps>([])
  const [ppm, setPpm] = useState<number[][][]>([])

  useEffect(() => {

    (async () => {
      const session = await ort.InferenceSession.create(
        '/deepsea.onnx', { executionProviders: ['wasm'] });
      setModel(session)
      setDownloading(false)
    })();

  }, [])

  const predict = async () => {
    if (model === null)
      return;
    else {
      const input = oneHotEncoder(dna);

      const results: any = await model.run({ arg0: input });

      const probs = new Float32Array(results.sigmoid.data);
      const gradCam = new Float32Array(results.div.data);

      let tfProbs: BarChartProps = [];

      for (let i = 0; i < probs.length; i++) {
        tfProbs[i] = { name: tfNames[i], prob: probs[i], index: i }
      }

      tfProbs = tfProbs.sort((a, b) => (a.prob > b.prob) ? -1 : 1)
      setProbs(tfProbs.slice(0, 25))

      let ppm: number[][][] = new Array(20).fill(null).map(() => []);

      const scale = 1.5;

      for (let i = 0; i < gradCam.length / 4; i++) {
        ppm[Math.floor(i / 50)].push(
          [
            gradCam[i] * scale,
            gradCam[i + 1000] * scale,
            gradCam[i + 2000] * scale,
            gradCam[i + 3000] * scale
          ]
        )
      }
      setPpm(ppm)
    }
  }

  return (
    <div className="flex flex-col items-center justify-center mt-10">
      <div>
        {downloading &&
          <>
            <div className="flex flex-end bg-blue-100 border-t border-b border-blue-500 text-blue-700 px-4 py-3 mb-15" role="alert">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-gray-900"></div>
              <p className="font-bold">Downloading the model...</p>
            </div>
          </>
        }
      </div>
      <div className="w-1/2">
        <label
          htmlFor="sequence"
          className="block overflow-hidden rounded-md border border-gray-200 px-3 py-2 shadow-sm focus-within:border-blue-600 focus-within:ring-1 focus-within:ring-blue-600"
        >
          <span className="text-xs font-medium text-gray-700"> DNA input (ATCG)</span>
          <textarea
            rows={12}
            placeholder="ACTCTT"
            className="mt-1 w-full border-none p-0 focus:border-transparent focus:outline-none focus:ring-0 text-lg"
            value={dna}
            onChange={(e) => setDna(e.target.value)}
          />
        </label>
        <div className='mt-1 flex justify-between'>
          <div className='flex gap-3'>
            {
              tfs.map((tf, index) =>
                <span key={index}
                  className='text-blue-500 cursor-pointer underline text-lg'
                  onClick={() => { setDna(tf.sequence) }}>
                  {tf.name}
                </span>
              )
            }
          </div>
          <div>
            <button
              className="bg-blue-500 hover:bg-blue-400 text-white font-bold py-2 px-4 border-b-4 border-blue-700 hover:border-blue-500 rounded"
              onClick={() => predict()}>
              Predict
            </button>
          </div>
        </div>
        <div className='mt-10'>
          <BarChart tfProbs={probs} />
        </div>
        <div>
          {
            ppm.length > 0 ? ([...Array(20).keys()]).map((i) =>
              <div key={i}>
                <DNALogo ppm={ppm[i]} />
              </div>
            ) : null
          }
        </div>
      </div>
    </div >
  )
}

export default App
