import csv
import numpy as np

import os
import matplotlib.pyplot as plt
from matplotlib import pyplot


if __name__ == '__main__':

    f_reward = open('/home/zcj/spiking-ddpg-mapless-navigation-master/writer/reward_1.csv', 'r', encoding='utf-8')
    reward_reader = csv.reader(f_reward)
    # loss_writer.writerow(["overall_steps", "actor_loss", "critic_loss"])
    headers_reward = next(reward_reader)
    r_steps = []
    avg_rewards = []
    for reward in reward_reader:
        r_steps.append(int(reward[0]))
        avg_rewards.append(np.clip(float(reward[1]),-1.2,100))

    f_loss = open('/home/zcj/spiking-ddpg-mapless-navigation-master/writer/loss_1.csv', 'r', encoding='utf-8')
    loss_reader = csv.reader(f_loss)
    # loss_writer.writerow(["overall_steps", "actor_loss", "critic_loss"])
    headers = next(loss_reader)
    steps=[]
    actor_loss=[]
    critic_loss=[]
    Qmean=[]
    Qcnt=0
    Qsum=0
    scnt=0
    for loss in loss_reader:
        st=int(loss[0])
        al = -float(loss[1])
        steps.append(st)
        actor_loss.append(al)

        '''
        if st <r_steps[Qcnt]:
            Qsum+=al
            scnt+=1
        else:
            if scnt!=0:
                v=Qsum/scnt
                Qmean.append(v)
            else:
                Qmean.append(0)
            Qsum=0
            scnt=0
            Qcnt+=1
        '''

        critic_loss.append(float(loss[2]))
        # print(loss)

    # _rewards=np.array(avg_rewards)
    # _rewards=np.clip(_rewards ,-0.2,5)
    # Rmean=[]
    # jup=5
    # rn=1000//jup
    # for i in range(rn):
    #     r = np.sum(_rewards[i*jup:i*jup+jup])/jup
    #     Rmean.append(r)

    plt.figure(1)
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 1, 2)

    plt.sca(ax1)
    plt.xlabel('steps')
    plt.ylabel('actor_loss')
    plt.plot(steps, actor_loss,'r-.')
    # plt.xlim(0,1)

    plt.sca(ax2)
    plt.xlabel('steps')
    plt.ylabel('critic_loss')
    plt.plot(steps, critic_loss, 'g--')

    plt.sca(ax3)
    plt.xlabel('steps')
    plt.ylabel('avg_reward')
    plt.plot(r_steps, avg_rewards, 'b--')

    plt.show()


    '''
    plt.style.use('seaborn-whitegrid')
    palette = pyplot.get_cmap('Set1')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    fig = plt.figure()

    color = palette(0)  # acc的颜色  1的话是蓝色
    ax = fig.subplots()

    iters = list(range(1000))
    iters5 = list(range(0,1000,jup))


    # ax.plot(iters[0:100], avg_rewards[0:100], color=palette(0), linewidth=2.0)
    # ax.plot(iters[100:300], avg_rewards[100:300], color=palette(1), linewidth=2.0)
    # ax.plot(iters[300:600], avg_rewards[300:600], color=palette(2), linewidth=2.0)
    # ax.plot(iters[600:1000], avg_rewards[600:1000], color=palette(3), linewidth=2.0)

    # ax.plot(iters5[0:100//jup], Rmean[0:100//jup], color=palette(0), linewidth=1.5,label='gazebo env1 (0~100 epoch)')
    # ax.plot(iters5[100//jup:300//jup], Rmean[100//jup:300//jup], color=palette(1), linewidth=1.5,label='gazebo env2 (100~300 epoch)')
    # ax.plot(iters5[300//jup:600//jup], Rmean[300//jup:600//jup], color=palette(2), linewidth=1.5,label='gazebo env3 (300~600 epoch)')
    # ax.plot(iters5[600//jup:1000//jup], Rmean[600//jup:1000//jup], color=palette(3), linewidth=1.5,label='gazebo env4 (600~1000 epoch)')
    # ax.plot(iters5, Rmean, color=color, linewidth=3.0)

    ax.plot(iters[0:100], Qmean[0:100], color=palette(0), linewidth=2,label='gazebo env1 (0~100 epoch)')
    ax.plot(iters[100:300], Qmean[100:300], color=palette(1), linewidth=2,label='gazebo env2 (100~300 epoch)')
    ax.plot(iters[300:600], Qmean[300:600], color=palette(2), linewidth=2,label='gazebo env3 (300~600 epoch)')
    ax.plot(iters[600:1000], Qmean[600:1000], color=palette(3), linewidth=2,label='gazebo env4 (600~1000 epoch)')

    # ax.plot(iters, Qmean, color=color, linewidth=2.0)
    # ax.plot(steps, actor_loss, color=color, linewidth=3.0)
    # ax.fill_between(iters, r1, r2, color=color, alpha=0.2)

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 12,
             }
    ax.legend(loc='lower right', prop=font2)
    ax.set_xlabel(' Epochs', fontsize=12)
    ax.set_ylabel('Mean Q value', fontsize=12)
    # ax.set_ylabel('average rewards', fontsize=12)
    # plt.ylim(-0.4, 1)
    plt.show()

    # fig = plt.figure()
    # iters = list(range(30))
    # root = r"F:\GitCode\Python\datasets"
    # re_acc = os.path.join(root, "eval_acces_new.pt")
    # color = palette(0)  # acc的颜色  1的话是蓝色
    # ax = fig.subplots()
    # acces_avg = np.mean(re_acc, axis=0) * 100
    # acces_std = np.std(re_acc, axis=0) * 100
    # r1 = list(map(lambda x: x[0] - x[1], zip(acces_avg, acces_std)))  # 上方差
    # r2 = list(map(lambda x: x[0] + x[1], zip(acces_avg, acces_std)))  # 下方差
    #
    # ax.plot(iters, acces_avg, color=color, linewidth=3.0)
    # ax.fill_between(iters, r1, r2, color=color, alpha=0.2)
    #
    # ax.legend(loc='lower right', prop=font1)
    # ax.set_xlabel('Epochs', fontsize=12)
    # ax.set_ylabel('Test accuracy (%)', fontsize=12)
    # plt.show()


    #9361
    '''