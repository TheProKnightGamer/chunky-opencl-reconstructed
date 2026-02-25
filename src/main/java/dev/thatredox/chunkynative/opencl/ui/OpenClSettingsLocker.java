package dev.thatredox.chunkynative.opencl.ui;

import javafx.application.Platform;
import javafx.scene.Node;
import javafx.scene.Parent;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Control;
import javafx.scene.control.Tooltip;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.ui.render.RenderControlsTab;

import java.util.*;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Wraps existing Chunky render control tabs to disable settings that are
 * not supported by the OpenCL GPU renderer.
 *
 * When the active renderer is "ChunkyClRenderer", the specified controls
 * are greyed out with a tooltip explaining they are unavailable on GPU.
 * When switching back to CPU, the controls are re-enabled immediately.
 */
public class OpenClSettingsLocker {
    public static final String OPENCL_RENDERER_ID = "ChunkyClRenderer";

    // fx:id values of controls to disable on the Lighting tab
    // (Importance sampling is now supported on GPU, so those controls are NOT locked.)
    private static final Set<String> LIGHTING_LOCKED_IDS = new HashSet<>(Arrays.asList(
    ));

    // fx:id values of controls to disable on the Advanced tab
    private static final Set<String> ADVANCED_LOCKED_IDS = new HashSet<>(Arrays.asList(
            "renderThreads",
            "cpuLoad",
            "fastFog",
            "octreeImplementation",
            "octreeSwitchImplementation",
            "bvhMethod",
            "biomeStructureImplementation"
    ));

    private static final String LOCK_TOOLTIP = "Not available with GPU (OpenCL) renderer";

    /** All active wrappers — notified when the renderer changes. */
    private static final List<LockedTabWrapper> allWrappers = new CopyOnWriteArrayList<>();

    /** Whether we have already hooked into the rendererSelect ChoiceBox. */
    private static volatile boolean rendererHooked = false;

    /** The scene reference for checking the current renderer. */
    private static Scene currentScene;

    /**
     * Called when the renderer selection changes. Updates lock state on ALL wrapped tabs.
     * @param selectedRenderer the newly selected renderer ID/name from the ChoiceBox
     */
    private static void onRendererChanged(String selectedRenderer) {
        if (selectedRenderer == null) return;
        boolean isOpenCl = OPENCL_RENDERER_ID.equals(selectedRenderer);
        Platform.runLater(() -> {
            for (LockedTabWrapper wrapper : allWrappers) {
                if (wrapper.lockedIds.isEmpty()) continue;
                wrapper.lastWasOpenCl = isOpenCl;
                wrapper.applyLockState(wrapper.delegate.getTabContent(), isOpenCl);
            }
        });
    }

    /**
     * Search a node tree for a ChoiceBox with fx:id "rendererSelect" and add a listener.
     */
    @SuppressWarnings("unchecked")
    private static void hookRendererSelect(Node root) {
        if (root == null || rendererHooked) return;
        if ("rendererSelect".equals(root.getId()) && root instanceof ChoiceBox) {
            rendererHooked = true;
            ((ChoiceBox<String>) root).getSelectionModel().selectedItemProperty().addListener(
                    (obs, oldVal, newVal) -> onRendererChanged(newVal));
            return;
        }
        if (root instanceof Parent) {
            for (Node child : ((Parent) root).getChildrenUnmodifiable()) {
                hookRendererSelect(child);
                if (rendererHooked) return;
            }
        }
    }

    /**
     * Wraps existing tabs to add OpenCL settings locking behavior.
     * The wrapper intercepts update() calls and disables/enables controls
     * based on the active renderer.
     */
    public static List<RenderControlsTab> wrapTabs(Collection<RenderControlsTab> tabs) {
        List<RenderControlsTab> result = new ArrayList<>();
        for (RenderControlsTab tab : tabs) {
            String title = tab.getTabTitle();
            if ("Lighting".equals(title)) {
                result.add(new LockedTabWrapper(tab, LIGHTING_LOCKED_IDS));
            } else if ("Advanced".equals(title)) {
                result.add(new LockedTabWrapper(tab, ADVANCED_LOCKED_IDS));
            } else {
                result.add(tab);
            }
        }
        return result;
    }

    /**
     * Wrapper around a RenderControlsTab that disables specific controls
     * when the OpenCL renderer is active.
     */
    private static class LockedTabWrapper implements RenderControlsTab {
        private final RenderControlsTab delegate;
        private final Set<String> lockedIds;
        private final Map<String, Tooltip> originalTooltips = new HashMap<>();
        private boolean lastWasOpenCl = false;

        LockedTabWrapper(RenderControlsTab delegate, Set<String> lockedIds) {
            this.delegate = delegate;
            this.lockedIds = lockedIds;
            allWrappers.add(this);
        }

        @Override
        public void update(Scene scene) {
            delegate.update(scene);
            currentScene = scene;
            boolean isOpenCl = OPENCL_RENDERER_ID.equals(scene.getRenderer());
            // Only traverse the node tree when the state changes
            if (isOpenCl != lastWasOpenCl) {
                lastWasOpenCl = isOpenCl;
                Platform.runLater(() -> applyLockState(delegate.getTabContent(), isOpenCl));
            }
        }

        @Override
        public String getTabTitle() {
            return delegate.getTabTitle();
        }

        @Override
        public Node getTabContent() {
            return delegate.getTabContent();
        }

        @Override
        public void onChunksLoaded() {
            delegate.onChunksLoaded();
        }

        @Override
        public void setController(se.llbit.chunky.ui.controller.RenderControlsFxController controller) {
            delegate.setController(controller);
            currentScene = controller.getRenderController().getSceneManager().getScene();
            // Try to find the rendererSelect ChoiceBox in this tab's content.
            // Only the Advanced tab has it, but it's safe to search all tabs.
            // Use a short delay to ensure the FXML content is fully initialized;
            // retry once if the first attempt doesn't find the ChoiceBox.
            Platform.runLater(() -> {
                hookRendererSelect(delegate.getTabContent());
                if (!rendererHooked) {
                    // Retry after a short delay in case the node tree wasn't ready
                    new Thread(() -> {
                        try { Thread.sleep(500); } catch (InterruptedException ignored) {}
                        Platform.runLater(() -> hookRendererSelect(delegate.getTabContent()));
                    }).start();
                }
            });
        }

        private void applyLockState(Node root, boolean lock) {
            if (root == null) return;
            applyLockStateRecursive(root, lock);
        }

        private void applyLockStateRecursive(Node node, boolean lock) {
            String id = node.getId();
            if (id != null && lockedIds.contains(id)) {
                node.setDisable(lock);
                if (lock) {
                    // Also disable parent HBox if the control is inside one (for label+choicebox combos)
                    Node parent = node.getParent();
                    if (parent instanceof javafx.scene.layout.HBox) {
                        parent.setDisable(true);
                    }
                    // Add tooltip to explain why it's locked
                    if (node instanceof Control) {
                        Control control = (Control) node;
                        originalTooltips.putIfAbsent(id, control.getTooltip());
                        control.setTooltip(new Tooltip(LOCK_TOOLTIP));
                    }
                } else {
                    Node parent = node.getParent();
                    if (parent instanceof javafx.scene.layout.HBox) {
                        parent.setDisable(false);
                    }
                    // Restore original tooltip
                    if (node instanceof Control && originalTooltips.containsKey(id)) {
                        ((Control) node).setTooltip(originalTooltips.get(id));
                    }
                }
                return; // Don't recurse into locked containers
            }

            // Recurse into children
            if (node instanceof Parent) {
                for (Node child : ((Parent) node).getChildrenUnmodifiable()) {
                    applyLockStateRecursive(child, lock);
                }
            }
        }
    }
}
