cat run_2022-07-12-14\:13\:25/*.log | grep loss_D | cut -d ':' -f 2 | cut -d ',' -f1 | sed s/' '//g > log.log

python log_show.py loss_D

cat run_2022-07-12-14\:13\:25/*.log | grep loss_G_fake | cut -d ':' -f 3 | sed s/','//g > log.log

python log_show.py loss_G_fake


cat run_2022-07-12-14\:13\:25/*.log | grep loss_perturb | cut -d ':' -f 2 | cut -d ',' -f1 | sed s/' '//g > log.log

python log_show.py loss_perturb


cat run_2022-07-12-14\:13\:25/*.log | grep loss_adv | cut -d ':' -f 3 | sed s/', '//g > log.log

python log_show.py loss_adv