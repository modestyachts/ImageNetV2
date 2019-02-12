#!/bin/bash

WNID_FILE='test_wnids.json'

#Date formatting is 'YYYY-MM-DD'
python flickr_search.py "../data/flickr_api_keys.json" \
                        --wnids "${WNID_FILE}" \
                        --max_images 200 \
                        --max_date_taken "2013-07-11"\
                        --max_date_uploaded "2013-07-11"\
                        --min_date_taken "2012-07-11"\
                        --min_date_uploaded "2012-07-11" \
