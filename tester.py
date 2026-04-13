from natsort import natsorted

files = [
    '3472_tb_pooler',
    '3473_tb_pooler',
    '3474_tb_pooler',
    '3475_tb_pooler',
    '3476_tb_pooler',
    '3477_tb_pooler',
    '3478_tb_pooler',
    '3479_tb_pooler',
    '347_tb_pooler',
    '3480_tb_pooler',
    '3481_tb_pooler',
    '3482_tb_pooler',
    '3483_tb_pooler',
    '3484_tb_pooler',
    '3485_tb_pooler',
    '3486_tb_pooler',
    '3487_tb_pooler',
    '3488_tb_pooler',
    '3489_tb_pooler',
    '348_tb_pooler',
    '3490_tb_pooler']

files_ordered = natsorted(files)
print(files_ordered)