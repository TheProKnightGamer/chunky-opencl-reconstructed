package dev.thatredox.chunkynative.util;

import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class NativeCleaner extends Thread {
    public static final NativeCleaner INSTANCE = new NativeCleaner("Chunky Native Cleaner");

    protected final ReferenceQueue<Object> cleanerQueue = new ReferenceQueue<>();
    protected final List<Cleaner> cleaners = Collections.synchronizedList(new ArrayList<>());

    public static class Cleaner extends PhantomReference<Object> {
        protected final Runnable action;
        protected volatile boolean cleaned = false;

        protected Cleaner(Object ref, ReferenceQueue<Object> q, Runnable action) {
            super(ref, q);
            this.action = action;
        }

        public void clean() {
            if (!cleaned) {
                cleaned = true;
                action.run();
            }
        }
    }

    public NativeCleaner(String name) {
        super(name);
        this.start();
    }

    public Cleaner register(Object ref, Runnable action) {
        Cleaner cleaner = new Cleaner(ref, cleanerQueue, action);
        this.cleaners.add(cleaner);
        return cleaner;
    }

    /**
     * Unregister a previously registered cleaner. This removes the cleaner from the
     * internal list so it does not retain a strong reference after being cleaned
     * manually (e.g. when native resources are closed explicitly).
     */
    public void unregister(Cleaner cleaner) {
        this.cleaners.remove(cleaner);
    }

    @Override
    public void run() {
        try {
            while (!interrupted()) {
                Reference<?> cleaner = cleanerQueue.remove();
                if (cleaner instanceof Cleaner) {
                    ((Cleaner) cleaner).clean();
                    this.cleaners.remove(cleaner);
                }
            }
        } catch (InterruptedException ignored) {
        }
    }
}
