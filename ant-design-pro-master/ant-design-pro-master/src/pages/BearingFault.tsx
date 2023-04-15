import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import {Card, theme, Input, Result, Empty,Spin,message } from 'antd';
import React from 'react';
import {useState} from 'react'
import {ModelTest} from "@/services/ant-design-pro/api";
import ModelResult from "@/components/Charts/ModelResult";
import {BulbOutlined,BarChartOutlined} from '@ant-design/icons';
const { TextArea } = Input;
/**
 * 每个单独的卡片，为了复用样式抽成了组件
 * @param param0
 * @returns
 */

const InfoCard: React.FC<{
  title: string;
  result:any;
  loading:boolean,
}> = ({ title,result,loading }) =>
{
  const { useToken } = theme;

  const { token } = useToken();

  return (
    <div
      style={{
        backgroundColor: token.colorBgContainer,
        boxShadow: token.boxShadow,
        borderRadius: '8px',
        fontSize: '14px',
        color: token.colorTextSecondary,
        lineHeight: '22px',
        padding: '16px 19px',
        minWidth: '220px',
        flex: 1,
      }}
    >
      <div
        style={{
          display: 'flex',
          gap: '4px',
          alignItems: 'center',
        }}
      >
        <div
          style={{
            fontSize: '16px',
            color: token.colorText,
            paddingBottom: 8,
          }}
        >
           <BulbOutlined />{title}
        </div>
      </div>
      <div
        style={{
          fontSize: '14px',
          color: token.colorTextSecondary,
          textAlign: 'center',
          lineHeight: '100px',
          marginBottom: 8,
          minHeight:'60px'
        }}
      >

        <Spin spinning={loading} size={"large"}>
          {
            result===""?<Empty/>:result==="正常数据"?<Result
    status="success"
    title={result}
    subTitle={"模型预测结果为正常数据。"}
  />:<Result
    title={result}
    subTitle={"模型预测结果为"+result+"。"}

  />
          }
        </Spin>



      </div>
    </div>
  );
};
const InfoCard1: React.FC<{
  title: string;
  result:any;
  data:any;
  loading:boolean
}> = ({ title  ,result,data,loading}) =>
{
  const { useToken } = theme;

  const { token } = useToken();

  return (
    <div
      style={{
        backgroundColor: token.colorBgContainer,
        boxShadow: token.boxShadow,
        borderRadius: '8px',
        fontSize: '14px',
        color: token.colorTextSecondary,
        lineHeight: '22px',
        padding: '16px 19px',
        minWidth: '220px',
        flex: 1,
      }}
    >
      <div
        style={{
          display: 'flex',
          gap: '4px',
          alignItems: 'center',
        }}
      >
        <div
          style={{
            fontSize: '16px',
            color: token.colorText,
            paddingBottom: 8,
          }}
        >
         <BarChartOutlined />{title}
        </div>
      </div>
      <div
        style={{
          fontSize: '14px',
          color: token.colorTextSecondary,
          textAlign: 'justify',
          lineHeight: '22px',
          marginBottom: 8,
        }}
      >
        <Spin spinning={loading} size={"large"}>
                  {result===""?<Empty/>:<ModelResult data={data}/>}

        </Spin>


      </div>
    </div>
  );
};

const BearingFault: React.FC = () => {
  const { token } = theme.useToken();
  const { initialState } = useModel('@@initialState');
  const [data, setData] = useState([]);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const handleSubmit = async (data: any) => {
    try {
      // 登录
      setLoading(true)
      const modelResult = await ModelTest( data );
      if (modelResult) {
        message.success("请求成功！")
        setLoading(false)
        setResult(modelResult.result)
        console.log("modelResult.data=",modelResult.data)
        setData(modelResult.data)

      }
    } catch (error) {
      setLoading(false)
      console.log("#######model error######",error);
    }
  };
  return (
    <PageContainer>
      <Card
        style={{
          borderRadius: 8,
        }}
        bodyStyle={{
          backgroundImage:
            initialState?.settings?.navTheme === 'realDark'
              ? 'background-image: linear-gradient(75deg, #1A1B1F 0%, #191C1F 100%)'
              : 'background-image: linear-gradient(75deg, #FBFDFF 0%, #F5F7FF 100%)',
        }}
      >
        <div
        >
          <div
            style={{
              fontSize: '20px',
              color: token.colorTextHeading,
              lineHeight:'50px'
            }}
          >
            请输入振动信号：
          </div>
          <TextArea onPressEnter={async (e)=>{
            await handleSubmit(e.target.value);

          }} autoSize={{ minRows: 3, maxRows: 5 }}/>
           <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 16,
              marginTop:"10px"
            }}
          >
            <InfoCard
              title=" 预测结果"
             result={result}
              loading={loading}



            />
            <InfoCard1
              title=" 模型预测概率"
              result={result}
              data={data}
              loading={loading}

            />
          </div>
        </div>
      </Card>
    </PageContainer>
  );
};

export default BearingFault;
