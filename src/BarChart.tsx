import { useEffect, useState } from 'react'
import ReactApexChart from 'react-apexcharts'
import Modal from './Modal'

type BarChartProp = {
    prob: number,
    name: string
}
type BarChartProps = Array<BarChartProp>

function barOptions(tf_probs: BarChartProps, setOpen: (open: boolean) => void, calculateLogo: (tfname: string) => void) {

    return {
        chart: {
            id: 'basic-bar',
            events: {
                dataPointSelection: (_: any, __: any, config: any) => {
                    setOpen(true)
                    calculateLogo(tf_probs[config.dataPointIndex].name)
                }
            }
        },
        xaxis: {
            categories: Array.isArray(tf_probs) ? tf_probs.map((tf: BarChartProp) => tf.name) : [],
            labels: {
                style: {
                    fontSize: "14px"
                }
            }
        },
        yaxis: {
            labels: {
                maxWidth: 300,
                style: {
                    fontSize: "20px"
                }
            }
        },
        plotOptions: {
            bar: {
                borderRadius: 4,
                horizontal: true,
            }
        },
    }
}

function barSeries(tf_probs: BarChartProps) {
    return [
        {
            name: 'series-1',
            data: Array.isArray(tf_probs) ? tf_probs.map((tf: BarChartProp) => tf.prob) : [],
        },
    ]
}

const BarChart = ({ tfProbs, predictLogo }: { tfProbs: BarChartProps, predictLogo: (tfname: string) => Promise<number[][]> }) => {

    const [isOpen, setIsOpen] = useState(false)

    const calculateLogo = async (tfname: string) => {
        const ppm = await predictLogo(tfname)
        setPpm(ppm)
    }

    const [options, setOptions] = useState(barOptions(tfProbs, setIsOpen, calculateLogo))
    const [series, setSeries] = useState(barSeries(tfProbs))

    const [ppm, setPpm] = useState<number[][] | null>(null)

    useEffect(() => {
        setOptions(barOptions(tfProbs, setIsOpen, calculateLogo))
        setSeries(barSeries(tfProbs))
    }, [tfProbs])

    return (
        <div>
            <ReactApexChart
                options={options}
                series={series}
                type='bar'
                height={750}
            />
            <Modal ppm={ppm || []} isOpen={isOpen} setIsOpen={setIsOpen} />
        </div>
    )
}

export type { BarChartProps }
export default BarChart