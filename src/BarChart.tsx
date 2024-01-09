import { useEffect, useState } from 'react'
import ReactApexChart from 'react-apexcharts'

type BarChartProp = {
    prob: number,
    name: string,
    index: number
}
type BarChartProps = Array<BarChartProp>

function barOptions(tf_probs: BarChartProps) {

    return {
        chart: {
            id: 'basic-bar',
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
                },
                align: 'left'
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

const BarChart = ({ tfProbs }: { tfProbs: BarChartProps }) => {

    const [options, setOptions] = useState(barOptions(tfProbs))
    const [series, setSeries] = useState(barSeries(tfProbs))

    useEffect(() => {
        setOptions(barOptions(tfProbs))
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
        </div>
    )
}

export type { BarChartProps }
export default BarChart