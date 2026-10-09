#!/bin/bash

(cd cases/hourly_restart && ../../../case_submit.sh)
(cd cases/monthly&& ../../../case_submit.sh)
(cd cases/nest_gfs && ../../../case_submit.sh)
(cd cases/sm_ctrl && ../../../case_submit.sh)
(cd cases/sm_dry && ../../../case_submit.sh)
(cd cases/tgrad_ctrl && ../../../case_submit.sh)
(cd cases/tgrad_sst4 && ../../../case_submit.sh)