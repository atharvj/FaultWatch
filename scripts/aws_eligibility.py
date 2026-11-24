"""Read the Free Tier plan state for an explicitly selected AWS profile."""

import argparse
import json

import boto3


def eligible(state):
    return state.get("accountPlanType") == "FREE" and state.get("accountPlanStatus") == "ACTIVE"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    session = boto3.Session(profile_name=args.profile)
    state = session.client("freetier", region_name="us-east-1").get_account_plan_state()
    print(
        json.dumps(
            {
                "plan_type": state.get("accountPlanType"),
                "status": state.get("accountPlanStatus"),
                "active_free_plan": eligible(state),
            }
        )
    )
    if not eligible(state):
        raise SystemExit(1)
