import { PageContainer } from '@ant-design/pro-components';
import { useModel } from '@umijs/max';
import { Card, theme } from 'antd';
import React from 'react';
import AccsLine from "@/components/Charts/AccsLine";
/**
 * 每个单独的卡片，为了复用样式抽成了组件
 * @param param0
 * @returns
 */
const InfoCard: React.FC<{
  title: string;
  index: number;
  desc: string;
  href: string;
}> = ({ title, href, index, desc }) => {
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
            width: 48,
            height: 48,
            lineHeight: '22px',
            backgroundSize: '100%',
            textAlign: 'center',
            padding: '8px 16px 16px 12px',
            color: '#FFF',
            fontWeight: 'bold',
            backgroundImage:
              "url('https://gw.alipayobjects.com/zos/bmw-prod/daaf8d50-8e6d-4251-905d-676a24ddfa12.svg')",
          }}
        >
          {index}
        </div>
        <div
          style={{
            fontSize: '16px',
            color: token.colorText,
            paddingBottom: 8,
          }}
        >
          {title}
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
        {desc}
      </div>
      <a href={href} target="_blank" rel="noreferrer">
        了解更多 {'>'}
      </a>
    </div>
  );
};

const Welcome: React.FC = () => {
  const { token } = theme.useToken();
  const { initialState } = useModel('@@initialState');
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
          style={{
            backgroundPosition: '100% -30%',
            backgroundRepeat: 'no-repeat',
            backgroundSize: '223px auto',
            backgroundImage:"url(" + require("../../public/icons/yinxing.png") + ")",

            // backgroundImage:
            //   "url('https://gw.alipayobjects.com/mdn/rms_a9745b/afts/img/A*BuFmQqsB2iAAAAAAAAAAAAAAARQnAQ')",
          }}
        >
          <div
            style={{
              fontSize: '20px',
              color: token.colorTextHeading,
            }}
          >
            欢迎使用基于SNN的故障诊断系统
          </div>
          <p
            style={{
              fontSize: '14px',
              color: token.colorTextSecondary,
              lineHeight: '22px',
              marginTop: 16,
              marginBottom: 32,
              width: '85%',
            }}
          >
            &emsp;&emsp;轴承作为旋转机械中的关键部件，实时监测旋转机械中轴承作业状态能够避免经济损失及安全事故的发生。<br/>
            &emsp;&emsp;本课题拟通过对故障诊断领域的前沿技术及相关文献进行分析，构建基于脉冲神经网络的旋转机械轴承故障诊断模型，并与主流的基于深度学习的轴承故障诊断模型进行性能比较与分析。
            <br/>
            &emsp;&emsp;此外，本课题拟基于主流Web开发框架，设计并实现基于脉冲神经网络的旋转轴承故障诊断系统，实时可视化诊断结果。要求识别效果准确，系统功能完善，系统稳定可靠。
          </p>
           <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: 16,
            }}
          >
            <InfoCard
              index={1}
              href="https://zhuanlan.zhihu.com/p/416187474"
              title="了解 SNN"
              desc="&emsp;&emsp;第三代神经网络，脉冲神经网络 (Spiking Neural Network，SNN) ，旨在弥合神经科学和机器学习之间的差距，使用最拟合生物神经元机制的模型来进行计算，更接近生物神经元机制。脉冲神经网络与目前流行的神经网络和机器学习方法有着根本上的不同。SNN 使用脉冲——这是一种发生在时间点上的离散事件——而非常见的连续值。每个峰值由代表生物过程的微分方程表示出来，其中最重要的是神经元的膜电位。本质上，一旦神经元达到了某一电位，脉冲就会出现，随后达到电位的神经元会被重置。对此，最常见的模型是 Leaky Integrate-And-Fire (LIF) 模型。此外，SNN 通常是稀疏连接的，并会利用特殊的网络拓扑。"
            />
            <InfoCard
              index={2}
              title="了解 故障诊断"
              href="https://baike.baidu.com/item/%E6%95%85%E9%9A%9C%E8%AF%8A%E6%96%AD/5928542"
              desc="&emsp;&emsp;利用各种检查和测试方法，发现系统和设备是否存在故障的过程是故障检测；而进一步确定故障所在大致部位的过程是故障定位。故障检测和故障定位同属网络生存性范畴。要求把故障定位到实施修理时可更换的产品层次（可更换单位）的过程称为故障隔离。故障诊断就是指故障检测和故障隔离的过程"
            />
          </div>
        </div>
      </Card>
    </PageContainer>
  );
};

export default Welcome;
