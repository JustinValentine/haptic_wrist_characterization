"""Zero calibration of the P4 wrist: home pose and single-turn encoder survey.

Run once, and again whenever something moves a joint relative to its motors
(a cable, a capstan, a puck). With the wrist exactly at HOME_DEG it records
each motor's absolute single-turn encoder (0x3012:01); then, while you move
each joint by hand, it measures which way each encoder counts relative to its
puck's position counter. The result, wrist_calibration.json, lets every later
run zero the wrist from anywhere within half a motor turn of home (PS within
~20 deg; FE and UR within |dFE| + |dUR| < ~24 deg) instead of exact placement,
as libbarrett-canopen's bt-wam-zerocal does on the WAM.

The calibration also zeroes the wrist (from the home reading), so the next
experiment keeps that zero if no puck is reset in between.

Run:  python zero_calibration.py [--sim] [--yes]
"""
import argparse
import json
import math
from datetime import datetime, timezone

import device_config
from experiment_common import save_zero_record, wrist_config, write_json

MIN_MOTION_COUNTS = 200   # ~18 deg of motor turn: enough to tell the direction


def wait(args, message):
    if args.yes:
        print(message)
        return
    input(message + ' Press Enter... ')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--sim', action='store_true', help='run against the simulated wrist')
    parser.add_argument('--yes', action='store_true', help='no prompts (simulation)')
    parser.add_argument('--interface', default=device_config.CAN_INTERFACE)
    parser.add_argument('--out', default=device_config.ZERO_CALIBRATION_FILE)
    args = parser.parse_args(argv)
    from wrist_backend import p4wrist

    plant = None
    if args.sim:
        import p4_sim
        bus, plant = p4_sim.make_sim(nodes=device_config.NODES)
        wrist = p4wrist.Wrist.callback(bus, wrist_config(p4wrist, True))
    else:
        wrist = p4wrist.Wrist.socketcan(args.interface, wrist_config(p4wrist, False))
    with wrist:
        _, report = wrist.connect(None)   # always reset: a fresh counter origin
        print(report)
        info = wrist.drive_info()
        cpr = [d['counts_per_rev'] for d in info]
        home = [math.radians(x) for x in device_config.HOME_DEG]
        wait(args, f'Place the wrist EXACTLY at the home pose {device_config.HOME_DEG} deg '
                   '(PS, FE, UR) and hold it still.')
        counts0, single0 = wrist.read_encoders()
        wrist.zero(p4wrist.zero_cal(home), exact=True)
        if not args.sim:
            save_zero_record(p4wrist, wrist.zero_record())
        print(f'Home: counters {counts0}, single-turn encoders {single0}')

        dirs = [0, 0, 0]
        for attempt in range(5):
            missing = [device_config.NODES[i] for i in range(3) if dirs[i] == 0]
            if not missing:
                break
            if args.sim:
                # The simulated operator turns every motor about a third of a turn.
                plant.m = [m + d for m, d in zip(plant.m, (2.1, -1.7, 2.4))]
                for i, puck in enumerate(bus.pucks):
                    puck.angle = plant.m[i]
            else:
                wait(args, f'Move the joints by hand (PS and FE or UR, 10-20 deg each, motors '
                           f'{missing} still unmeasured), then hold the wrist still.')
            counts1, single1 = wrist.read_encoders()
            for i in range(3):
                moved = counts1[i] - counts0[i]
                if dirs[i] == 0 and abs(moved) >= MIN_MOTION_COUNTS:
                    dirs[i] = p4wrist.single_turn_direction(moved, single1[i] - single0[i], cpr[i])
            print(f'Directions so far (motors {list(device_config.NODES)}): {dirs}')
        if not all(dirs):
            raise SystemExit('Could not measure every encoder direction; move each motor '
                             f'at least {MIN_MOTION_COUNTS} counts (about '
                             f'{360 * MIN_MOTION_COUNTS / cpr[0]:.0f} deg of motor turn).')

        calibration = {
            'home_deg': list(device_config.HOME_DEG),
            'single_turn_at_home': single0,
            'single_turn_dir': dirs,
            'counts_per_rev': cpr,
            'nodes': list(device_config.NODES),
            'j2mp': [list(r) for r in device_config.J2MP],
            'recorded_utc': datetime.now(timezone.utc).isoformat(),
            'simulated': args.sim,
        }
        if args.sim and args.out == device_config.ZERO_CALIBRATION_FILE:
            print(json.dumps(calibration, indent=2))
            return calibration
        write_json(args.out, calibration)
        print(f'Wrote {args.out}. Later runs zero from the single-turn encoders when the '
              'wrist is placed within half a motor turn of home.')
        return calibration


if __name__ == '__main__':
    main()
