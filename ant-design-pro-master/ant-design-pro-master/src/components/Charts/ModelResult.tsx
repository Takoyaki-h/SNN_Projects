import { useEffect, useRef } from "react";

import * as echarts from "echarts";

function ModelResult(data:any) {
  console.log("#########data#####",data)
  let test=data.data

  const chartRef = useRef(null);



  useEffect(() => {

    let chartInstance = echarts.init(chartRef.current);

    const option = {
  tooltip: {
    trigger: 'axis',
    axisPointer: {
      type: 'shadow'
    }
  },
  grid: {
    // left: '3%',
    // height:'80%',
    right: '0%',
    // bottom: '3%',
    top:'2%',
    containLabel: true
  },
  xAxis: [
    {
      type: 'category',
      data: ['正常数据', '内圈故障', '外圈故障', '滚动体故障'],
      axisTick: {
        alignWithLabel: true
      }
    }
  ],
  yAxis: [
    {
      type: 'value'
    }
  ],
  series: [
    {
      name: '概率',
      type: 'bar',
      // barWidth: '60%',
      data:test
    }
  ]
};





;

    chartInstance.setOption(option);

  }, [test]);



  return (

    <div style={{ textAlign: "center" }}>

      {/*<h2>振动信号可视化</h2>*/}

      <div ref={chartRef} style={{ height: "400px" }}></div>

    </div>

  );

}



export default ModelResult;
