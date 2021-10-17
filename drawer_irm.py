import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

# erm_acc = np.load('acc_ERM.npy')
# erm_loss = np.load('loss_ERM.npy')[:1700]
# erm_p = np.load('penalty_ERM.npy')[:1700]

irm_loss = np.load('nll_IRM_ColoredMNIST.npy')[100:1900]

irm_p = np.load('penalty_IRM_ColoredMNIST.npy')[100:1900]
irm_acc = np.load('acc_IRM_ColoredMNIST.npy')
irm_t = np.load('trm_IRM_ColoredMNIST.npy')[100:1900]
trm_loss = np.load('nll_TRM_ColoredMNIST.npy')[100:1900]
trm_p = np.load('penalty_TRM_ColoredMNIST.npy')[100:1900]
trm_acc = np.load('acc_TRM_ColoredMNIST.npy')
trm_t = np.load('trm_TRM_ColoredMNIST.npy')[100:1900]

irm_loss_pacs = np.load('nll_IRM_PACS.npy')[100:1900]
irm_p_pacs = np.load('penalty_IRM_PACS.npy')[100:1900]
irm_t_pacs = np.load('trm_IRM_PACS.npy')[100:1900]
trm_loss_pacs = np.load('nll_TRM_PACS.npy')[100:1900]
trm_p_pacs = np.load('penalty_TRM_PACS.npy')[100:1900]
trm_t_pacs = np.load('trm_TRM_PACS.npy')[100:1900]

mask = trm_p > 1
trm_p[mask] = trm_p[mask]/200
# print("nll:", irm_loss, trm_loss)
# print("p:", irm_p, trm_p)

irm_irmv1 = irm_loss + irm_p
trm_irmv1 = trm_loss + trm_p
irm_irmv1_pacs = irm_loss_pacs + irm_p_pacs * 0.1
trm_irmv1_pacs = trm_loss_pacs + trm_p_pacs * 0.1

step = np.arange(len(irm_loss))
step_acc = np.arange(len(irm_acc)) * 100
print(len(step_acc), len(step))

sns.set(style='whitegrid', font_scale=1.7,rc={"lines.linewidth": 3})
EMA_SPAN = 200
fig, ax1 = plt.subplots()
p1 = ax1.plot(trm_irmv1, alpha=0.3)[0]
mis_smooth_1 = pd.Series(trm_irmv1).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_1, c=p1.get_color(), label='TRM (C)')

EMA_SPAN = 200
p2 = ax1.plot(irm_irmv1, alpha=0.3, c=p1.get_color(),linestyle='dashed')[0]
mis_smooth_2 = pd.Series(irm_irmv1).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_2, c=p1.get_color(), label='IRMv1 (C)',linestyle='dashed')

EMA_SPAN = 200
p3 = ax1.plot(trm_irmv1_pacs, alpha=0.3)[0]
mis_smooth_3 = pd.Series(trm_irmv1_pacs).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_3, c=p3.get_color(), label='TRM (P)')

EMA_SPAN = 200
p2 = ax1.plot(irm_irmv1_pacs, alpha=0.3, c=p3.get_color(),linestyle='dashed')[0]
mis_smooth_4 = pd.Series(irm_irmv1_pacs).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_4, c=p3.get_color(), label='IRMv1 (P)',linestyle='dashed')

# ax2.plot(step_acc, trm_acc,linestyle='dashed', label='TRM (acc)')
# ax2.plot(step_acc, irm_acc,linestyle='dashed', label='IRM (acc)')
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc='upper right')

axins = ax1.inset_axes([0.7, 0.15, 0.2, 0.2])
axins.plot(trm_p, alpha=0.3)[0]
axins.plot(irm_p, alpha=0.3)[0]

axins.plot(mis_smooth_1, c=p1.get_color())
axins.plot(mis_smooth_2, c=p1.get_color(),linestyle='dashed')
axins.plot(mis_smooth_3, c=p3.get_color())
axins.plot(mis_smooth_4, c=p3.get_color(),linestyle='dashed')
ax1.legend()
# sub region of the original image
x1, x2, y1, y2 = 1500, 1800, 0, 0.5
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
# axins.set_yticklabels('')
ax1.indicate_inset_zoom(axins, edgecolor="black")


ax1.set_xlabel('Iteration')
ax1.set_ylabel('IRMv1 loss')
# ax2.set_ylabel('Accuracy')

plt.savefig('irmv1_loss_2.pdf', dpi=300, bbox_inches='tight')
plt.show()
plt.clf()


step = np.arange(len(trm_t))
step_acc = np.arange(len(trm_acc)) * 100
print(len(step_acc), len(step))
sns.set(style='whitegrid', font_scale=1.7,rc={"lines.linewidth": 3})
fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
EMA_SPAN = 200
p1 = ax1.plot(trm_t, alpha=0.3)[0]
mis_smooth_1 = pd.Series(trm_t).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_1, c=p1.get_color(), label='TRM (C)')

EMA_SPAN = 200
p2 = ax1.plot(irm_t, alpha=0.3,linestyle='dashed',c=p1.get_color())[0]
mis_smooth_2= pd.Series(irm_t).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_2, c=p1.get_color(), label='IRMv1 (C)',linestyle='dashed')

EMA_SPAN = 200
p3 = ax1.plot(trm_t_pacs, alpha=0.3)[0]
mis_smooth_3 = pd.Series(trm_t_pacs).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_3, c=p3.get_color(), label='TRM (P)')

EMA_SPAN = 200
p4 = ax1.plot(irm_t_pacs, alpha=0.3, c=p3.get_color(),linestyle='dashed')[0]
mis_smooth_4 = pd.Series(irm_t_pacs).ewm(span=EMA_SPAN).mean()
ax1.plot(mis_smooth_4, c=p3.get_color(), label='IRMv1 (P)',linestyle='dashed')

# ax2.plot(step_acc, irm_acc, linestyle='dashed', label='ERM (acc)')
# ax2.plot(step_acc, trm_acc, linestyle='dashed', label='TRM (acc)')

lines, labels = ax1.get_legend_handles_labels()
axins = ax1.inset_axes([0.7, 0.15, 0.2, 0.2])
axins.plot(trm_t, alpha=0.3)[0]
axins.plot(irm_t, alpha=0.3)[0]

axins.plot(mis_smooth_1, c=p1.get_color())
axins.plot(mis_smooth_2, c=p2.get_color(),linestyle='dashed')
ax1.legend()
# sub region of the original image
x1, x2, y1, y2 = 1500, 1800, 0, 0.2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
# axins.set_yticklabels('')
ax1.indicate_inset_zoom(axins, edgecolor="black")

# ax2.legend()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Transfer risk')
# ax2.set_ylabel('Accuracy')
plt.savefig('transfer_risk_2.pdf', dpi=300, bbox_inches='tight')
plt.show()

