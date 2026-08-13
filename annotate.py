import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import lasair

import config
from killonova_filter.kn_linear import classify_object_info

BATCH_SIZE = int(os.environ.get("LASAIR_BATCH_SIZE", "32"))
BATCH_WAIT_S = float(os.environ.get("LASAIR_BATCH_WAIT_S", "2"))
MAX_WORKERS = int(os.environ.get("LASAIR_MAX_WORKERS", "4"))
MAX_ALERT = int(os.environ.get("LASAIR_MAX_ALERT", "5"))
POLL_TIMEOUT_S = float(os.environ.get("LASAIR_POLL_TIMEOUT_S", "1"))


def collect_alert_batch(consumer, max_batch, max_wait_s, poll_timeout_s=POLL_TIMEOUT_S):
    """Collect objectIds from Kafka until batch size or wait time is reached."""
    object_ids = []
    deadline = time.time() + max_wait_s

    while len(object_ids) < max_batch and time.time() < deadline:
        msg = consumer.poll(timeout=poll_timeout_s)
        if msg is None:
            continue
        if msg.error():
            print(str(msg.error()))
            break

        jsonmsg = json.loads(msg.value())
        object_ids.append(jsonmsg["objectId"])

    return object_ids


def handle_batch(object_ids, lasair_client, topic_out, executor):
    """Fetch a batch from Lasair, classify in parallel, annotate results."""
    if not object_ids:
        return 0, 0

    object_infos = lasair_client.objects(object_ids)
    if not object_infos:
        return len(object_ids), 0

    results = list(executor.map(classify_object_info, object_infos))

    n_annotate = 0
    for object_id, result in zip(object_ids, results):
        lasair_client.annotate(
            topic_out,
            object_id,
            result["classification"],
            version=result.get("version", "0.1"),
            explanation=result["explanation"],
            classdict=result.get("classdict", {}),
            url=result.get("url", ""),
        )
        print(object_id, "-- annotated!")
        n_annotate += 1

    return len(object_ids), n_annotate


def main():
    group_id = config.GROUP_ID
    topic_in = config.TOPIC_IN
    topic_out = config.TOPIC_OUT

    consumer = lasair.lasair_consumer(
        "lasair-lsst-kafka_pub.lsst.ac.uk:9092", group_id, topic_in
    )
    lasair_client = lasair.lasair_client(config.LASAIR_API_TOKEN)

    n_alert = n_annotate = 0
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while n_alert < MAX_ALERT:
            remaining = MAX_ALERT - n_alert
            batch_size = min(BATCH_SIZE, remaining)
            object_ids = collect_alert_batch(
                consumer, batch_size, BATCH_WAIT_S, poll_timeout_s=POLL_TIMEOUT_S
            )
            if not object_ids:
                break

            batch_alert, batch_annotate = handle_batch(
                object_ids, lasair_client, topic_out, executor
            )
            n_alert += batch_alert
            n_annotate += batch_annotate

    print("Annotated %d of %d objects" % (n_annotate, n_alert))


if __name__ == "__main__":
    main()
