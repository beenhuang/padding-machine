# 
# run simulator, produced dataset/label and save to the pickle file
#
# simulate's all arguments:
# simulate.py --tc <~/client-traces> --tr <~/fakerelay-traces> \
#             --mc <machines/*-mc.c> --mr <machines/*-mr.c> \
#             --tor <~/tor-0.4.7.8> \
#             --save <ds-*.pkl> \ 
#             --worker <num:8> \
#             --class <num:50> --part <num:10> --sample <num:20> \
#             --length <maxnum:5000> 
#
#

# spring machines
#simulation/simulate.py --tc dataset/standard/client-traces/ \
#                       --tr dataset/standard/fakerelay-traces/ \
#                       --mc machines/spring-mc.c --mr machines/spring-mr.c \
#                       --tor ../tor-0.4.7.8 \
#                       --save ds-spring.pkl  

# interspace machines
#simulation/simulate.py --tc dataset/standard/client-traces/ \
#                       --tr dataset/standard/fakerelay-traces/ \
#                       --mc machines/interspace-mc.c --mr machines/interspace-mr.c \
#                       --tor ../tor-0.4.7.8 \
#                       --save ds-inter.pkl


# july machines
#simulation/simulate.py --tc dataset/standard/client-traces/ \
#                       --tr dataset/standard/fakerelay-traces/ \
#                       --mc machines/july-mc.c --mr machines/july-mr.c \
#                       --tor ../tor-0.4.7.8 \
#                       --save ds-july.pkl


# august machines
simulation/simulate.py --tc client-traces --tr fakerelay-traces \
                       --mc august-mc.c   --mr august-mr.c \
                       --tor tor-0.4.7.8  --out ds-august-01.pkl
                                     