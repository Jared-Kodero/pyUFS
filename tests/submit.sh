#!/bin/bash
# (cd cases/monthly && ../../../case_submit.sh)
# (cd cases/hourly_restart && ../../../case_submit.sh)
# (cd cases/nest_gfs && ../../../case_submit.sh)
# (cd cases/sm_ctrl && ../../../case_submit.sh)
# (cd cases/sm_dry && ../../../case_submit.sh)

(cd cases/pamip_ctrl && ../../../case_submit.sh)
(cd cases/pamip_futarcsic && ../../../case_submit.sh)
(cd cases/pamip_piarcsic && ../../../case_submit.sh)
(cd cases/pamip_native && ../../../case_submit.sh)