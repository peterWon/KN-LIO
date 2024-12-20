echo Runing PIN-LIO on VIRAL

# eee_01 eee_02 eee_03 nya_01 nya_02 nya_03 sbs_01 sbs_02 sbs_03 rtp_01 rtp_02 rtp_03 spms_01 spms_02 spms_03 tnp_01 tnp_02 tnp_03
for seq in spms_01 spms_03; do
    python3 pin_slam.py config/viral/${seq}.yaml
    echo Processed ${seq}.
done

