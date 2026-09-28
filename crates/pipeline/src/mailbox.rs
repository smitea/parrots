/// Bounded mailbox: Partial may be dropped, Final must be delivered
/// (backpressure policy)
#[derive(Debug, Clone)]
pub enum Mail<T> {
    Partial(T),
    Final(T),
}

pub async fn send_bounded<T>(
    tx: &tokio::sync::mpsc::Sender<Mail<T>>,
    mut msg: Mail<T>,
    dropped: &mut u64,
) {
    let is_final = matches!(msg, Mail::Final(_));
    loop {
        match tx.try_send(msg) {
            Ok(()) => return,
            Err(tokio::sync::mpsc::error::TrySendError::Full(back)) => {
                if is_final {
                    tokio::task::yield_now().await; // Final: wait for the consumer to free a slot
                    msg = back;
                } else {
                    *dropped += 1; // Partial: drop it
                    return;
                }
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(back)) => {
                let _ = back;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn partials_dropped_finals_delivered() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let mut drops = 0;
        send_bounded(&tx, Mail::Partial(1), &mut drops).await; // fills capacity
        send_bounded(&tx, Mail::Partial(2), &mut drops).await; // dropped immediately
        assert_eq!(drops, 1);
        let consumer = tokio::spawn(async move {
            let first = rx.recv().await.unwrap();
            let second = rx.recv().await.unwrap();
            (first, second)
        });
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        send_bounded(&tx, Mail::Final(3), &mut drops).await; // delivered after the consumer frees a slot
        drop(tx);
        let (first, second) = consumer.await.unwrap();
        let vals = [first, second].into_iter().map(|m| match m {
            Mail::Partial(v) | Mail::Final(v) => v,
        });
        assert_eq!(vals.collect::<Vec<_>>(), vec![1, 3]);
        assert_eq!(drops, 1);
    }
}
