#!/bin/sh
set -e

sed -i \
  -e "/^# WebArena rootless Docker tuning$/d" \
  -e "/^puma\\[worker_processes\\]/d" \
  -e "/^puma\\['worker_processes'\\]/d" \
  -e "/^sidekiq\\[max_concurrency\\]/d" \
  -e "/^sidekiq\\['max_concurrency'\\]/d" \
  /etc/gitlab/gitlab.rb

cat >> /etc/gitlab/gitlab.rb <<'EOF'

# WebArena rootless Docker tuning
puma['worker_processes'] = 2
EOF

gitlab-ctl reconfigure
gitlab-ctl stop puma || true
rm -rf /dev/shm/gitlab/puma/* 2>/dev/null || true
gitlab-ctl start puma
