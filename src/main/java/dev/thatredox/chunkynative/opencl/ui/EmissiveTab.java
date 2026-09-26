package dev.thatredox.chunkynative.opencl.ui;

import dev.thatredox.chunkynative.common.emissive.*;
import dev.thatredox.chunkynative.common.emissive.EmissiveSettings.BlockSetting;
import dev.thatredox.chunkynative.common.emissive.EmissiveSettings.Mode;
import javafx.application.Platform;
import javafx.beans.property.ReadOnlyObjectWrapper;
import javafx.beans.property.ReadOnlyStringWrapper;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.collections.transformation.FilteredList;
import javafx.geometry.Insets;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;
import javafx.stage.DirectoryChooser;
import javafx.stage.FileChooser;
import javafx.stage.Window;
import se.llbit.chunky.block.Block;
import se.llbit.chunky.renderer.ResetReason;
import se.llbit.chunky.renderer.scene.Scene;
import se.llbit.chunky.ui.render.RenderControlsTab;
import se.llbit.json.JsonValue;
import se.llbit.log.Log;

import java.io.File;
import java.lang.reflect.Method;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Lets the user pick which blocks glow, and whether a whole block glows or only the
 * texels its emission map (LabPBR {@code _s.png} alpha or OptiFine {@code _e.png})
 * marks. Packs are global; block choices are saved with the scene.
 */
public class EmissiveTab implements RenderControlsTab {
    private final VBox box = new VBox(8.0);
    private Scene scene;

    private final CheckBox enabled = new CheckBox("Use emission maps and the settings below");
    private final ObservableList<File> packs = FXCollections.observableArrayList();
    private final ListView<File> packList = new ListView<>(packs);
    private final CheckBox useChunkyPacks = new CheckBox("Also read Chunky's resource packs");
    private final Label packStatus = new Label();
    private final TextField filter = new TextField();
    private final CheckBox onlyRelevant = new CheckBox("Only blocks that glow or have a map");
    private final ObservableList<Row> rows = FXCollections.observableArrayList();
    private final FilteredList<Row> filteredRows = new FilteredList<>(rows);
    private final TableView<Row> table = new TableView<>(filteredRows);

    private final ExecutorService worker = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "ChunkyCL-Emissive");
        t.setDaemon(true);
        return t;
    });
    private Map<String, EmissionMask> masks = Collections.emptyMap();
    private boolean masksLoaded = false;
    private List<Block> shownPalette = null;
    private int shownPaletteSize = -1;
    private JsonValue shownSettings = null;
    private boolean updating = false;

    /** One block id in the scene. */
    private static final class Row {
        final String name;
        final boolean hasMap;
        final float chunkyEmittance;
        Mode mode;
        float strength;

        Row(String name, boolean hasMap, float chunkyEmittance, BlockSetting setting) {
            this.name = name;
            this.hasMap = hasMap;
            this.chunkyEmittance = chunkyEmittance;
            this.mode = setting.mode;
            this.strength = setting.strength;
        }

        /** What the renderer will do, in words. */
        String result() {
            boolean hasStrength = !Float.isNaN(strength);
            float base = hasStrength ? strength : (chunkyEmittance > 0 ? chunkyEmittance : 1);
            switch (mode) {
                case OFF:
                    return "no light";
                case WHOLE:
                    return String.format("whole block × %.2f", base);
                case MAP:
                    return String.format("%s × %.2f", hasMap ? "map" : "whole block (no map)", base);
                default:
                    if (chunkyEmittance <= 0) return "no light";
                    base = hasStrength ? strength : chunkyEmittance;
                    return String.format("%s × %.2f", hasMap ? "map" : "whole block", base);
            }
        }
    }

    public EmissiveTab(Scene scene) {
        this.scene = scene;
        box.setPadding(new Insets(10.0));

        enabled.setTooltip(new Tooltip("Off: every light-emitting block glows as a whole, as in Chunky."));
        enabled.setOnAction(e -> {
            if (updating) return;
            EmissiveSettings.setEnabled(this.scene, enabled.isSelected());
            shownSettings = this.scene.getAdditionalData(EmissiveSettings.SCENE_KEY);
            table.refresh();
            refreshScene();
        });

        // ---- packs ----
        Label packsTitle = new Label("Emission packs (highest priority first)");
        packsTitle.setStyle("-fx-font-weight: bold;");
        packList.setPrefHeight(90);
        packList.setCellFactory(lv -> new ListCell<File>() {
            @Override
            protected void updateItem(File item, boolean empty) {
                super.updateItem(item, empty);
                setText(empty || item == null ? null : item.getName());
                setTooltip(empty || item == null ? null : new Tooltip(item.getAbsolutePath()));
            }
        });
        Button addZip = new Button("Add pack…");
        addZip.setOnAction(e -> addZipPacks());
        Button addDir = new Button("Add folder…");
        addDir.setOnAction(e -> addFolderPack());
        Button remove = new Button("Remove");
        remove.setOnAction(e -> {
            File f = packList.getSelectionModel().getSelectedItem();
            if (f != null) {
                packs.remove(f);
                savePacks();
            }
        });
        Button up = new Button("Up");
        up.setOnAction(e -> movePack(-1));
        Button down = new Button("Down");
        down.setOnAction(e -> movePack(1));
        Button rescan = new Button("Rescan");
        rescan.setTooltip(new Tooltip("Convert the packs again, e.g. after editing an unzipped pack."));
        rescan.setOnAction(e -> loadMasks(true, true));
        FlowPane packButtons = new FlowPane(6, 6, addZip, addDir, remove, up, down, rescan);
        packButtons.setPrefWrapLength(280);

        useChunkyPacks.setSelected(EmissiveSettings.useChunkyPacks());
        useChunkyPacks.setOnAction(e -> {
            EmissiveSettings.setUseChunkyPacks(useChunkyPacks.isSelected());
            loadMasks(false, true);
        });
        wrap(packStatus);

        Label formats = new Label("LabPBR packs (emission in the alpha of _s.png) and OptiFine emissive "
                + "packs (_e.png) work. A pack's maps are converted once and cached, so later loads are fast. "
                + "A map is scaled to the resolution of the texture Chunky uses.");
        wrap(formats);

        // ---- blocks ----
        Label blocksTitle = new Label("Blocks in this scene");
        blocksTitle.setStyle("-fx-font-weight: bold;");
        filter.setPromptText("Filter blocks");
        filter.setMaxWidth(Double.MAX_VALUE);
        filter.textProperty().addListener((o, a, b) -> updateFilter());
        onlyRelevant.setSelected(true);
        onlyRelevant.setOnAction(e -> updateFilter());

        buildTable();

        Button allMaps = new Button("Use maps for all");
        allMaps.setTooltip(new Tooltip("Every block with an emission map glows where its map says."));
        allMaps.setOnAction(e -> {
            Map<String, BlockSetting> changes = new HashMap<>();
            for (Row r : rows) {
                if (r.hasMap && r.mode != Mode.MAP) {
                    r.mode = Mode.MAP;
                    changes.put(r.name, new BlockSetting(r.mode, r.strength));
                }
            }
            commit(changes);
        });
        Button reset = new Button("Reset all");
        reset.setOnAction(e -> {
            for (Row r : rows) {
                r.mode = Mode.DEFAULT;
                r.strength = Float.NaN;
            }
            EmissiveSettings.clearBlocks(this.scene);
            shownSettings = this.scene.getAdditionalData(EmissiveSettings.SCENE_KEY);
            table.refresh();
            refreshScene();
        });
        Button reload = new Button("Refresh list");
        reload.setOnAction(e -> rebuildRows());
        FlowPane blockButtons = new FlowPane(6, 6, allMaps, reset, reload);
        blockButtons.setPrefWrapLength(280);

        Label help = new Label("Default: blocks Chunky treats as light sources glow only where their map "
                + "says (or whole, without a map). Emission map: glow only where the map says. "
                + "Whole block: Chunky's behaviour. Off: never glow. Strength overrides the emittance "
                + "(blank = the Materials tab value, or 1 for blocks Chunky does not light). "
                + "Chunky samples lights only from blocks it treats as light sources, so a block "
                + "switched on here lights its surroundings through path tracing alone (noisier).");
        wrap(help);
        packList.setPrefWidth(200);
        packList.setMaxWidth(Double.MAX_VALUE);
        filter.setPrefWidth(200);

        box.getChildren().addAll(enabled, new Separator(),
                packsTitle, packList, packButtons, useChunkyPacks, packStatus, formats, new Separator(),
                blocksTitle, filter, onlyRelevant, table, blockButtons, help);

        packs.setAll(EmissiveSettings.extraPacks());
        syncFromScene(true);
        loadMasks(false, false);
    }

    /**
     * Let a long text wrap to the panel's width instead of setting it: the tool pane
     * sizes every tab to its content's preferred width.
     */
    private static void wrap(Labeled label) {
        label.setWrapText(true);
        label.setPrefWidth(100);
        label.setMaxWidth(Double.MAX_VALUE);
    }

    private void buildTable() {
        table.setPrefHeight(360);
        table.setPrefWidth(300);
        table.setMaxWidth(Double.MAX_VALUE);
        table.setEditable(true);
        table.setColumnResizePolicy(TableView.CONSTRAINED_RESIZE_POLICY);
        table.setPlaceholder(new Label("Load chunks to list their blocks."));
        table.setRowFactory(tv -> new TableRow<Row>() {
            @Override
            protected void updateItem(Row item, boolean empty) {
                super.updateItem(item, empty);
                setTooltip(empty || item == null ? null : new Tooltip(item.name
                        + (item.chunkyEmittance > 0
                            ? String.format("%nChunky emittance: %.2f", item.chunkyEmittance)
                            : "\nNot a light source in Chunky")
                        + (item.hasMap ? "\nHas an emission map" : "\nNo emission map")));
            }
        });

        TableColumn<Row, String> name = new TableColumn<>("Block");
        name.setCellValueFactory(c -> new ReadOnlyStringWrapper(
                c.getValue().name.startsWith("minecraft:") ? c.getValue().name.substring(10) : c.getValue().name));
        name.setPrefWidth(110);
        name.setMinWidth(60);

        TableColumn<Row, String> map = new TableColumn<>("Map");
        map.setCellValueFactory(c -> new ReadOnlyStringWrapper(c.getValue().hasMap ? "✓" : ""));
        map.setMinWidth(38);
        map.setPrefWidth(38);
        map.setMaxWidth(38);
        map.setStyle("-fx-alignment: CENTER;");

        TableColumn<Row, Mode> mode = new TableColumn<>("Mode");
        mode.setCellValueFactory(c -> new ReadOnlyObjectWrapper<>(c.getValue().mode));
        mode.setCellFactory(col -> new TableCell<Row, Mode>() {
            private final ChoiceBox<Mode> choice = new ChoiceBox<>(FXCollections.observableArrayList(Mode.values()));
            private boolean setting = false;

            {
                choice.setMaxWidth(Double.MAX_VALUE);
                choice.valueProperty().addListener((o, a, b) -> {
                    Row r = getTableRow() != null ? getTableRow().getItem() : null;
                    if (setting || r == null || b == null || b == r.mode) return;
                    r.mode = b;
                    commit(Collections.singletonMap(r.name, new BlockSetting(r.mode, r.strength)));
                });
            }

            @Override
            protected void updateItem(Mode item, boolean empty) {
                super.updateItem(item, empty);
                if (empty || item == null) {
                    setGraphic(null);
                } else {
                    setting = true;
                    choice.setValue(item);
                    setting = false;
                    setGraphic(choice);
                }
            }
        });
        mode.setMinWidth(112);
        mode.setPrefWidth(116);

        TableColumn<Row, String> strength = new TableColumn<>("Strength");
        strength.setCellValueFactory(c -> new ReadOnlyStringWrapper(
                Float.isNaN(c.getValue().strength) ? "" : Float.toString(c.getValue().strength)));
        strength.setCellFactory(col -> new TableCell<Row, String>() {
            private final TextField field = new TextField();

            {
                field.setPromptText("auto");
                field.setOnAction(e -> apply());
                field.focusedProperty().addListener((o, was, now) -> {
                    if (!now) apply();
                });
            }

            private void apply() {
                Row r = getTableRow() != null ? getTableRow().getItem() : null;
                if (r == null) return;
                String text = field.getText().trim();
                float value;
                if (text.isEmpty()) {
                    value = Float.NaN;
                } else {
                    try {
                        value = Math.max(0, Float.parseFloat(text));
                    } catch (NumberFormatException ex) {
                        field.setText(Float.isNaN(r.strength) ? "" : Float.toString(r.strength));
                        return;
                    }
                }
                if (Float.compare(value, r.strength) == 0) return;
                r.strength = value;
                commit(Collections.singletonMap(r.name, new BlockSetting(r.mode, r.strength)));
            }

            @Override
            protected void updateItem(String item, boolean empty) {
                super.updateItem(item, empty);
                if (empty) {
                    setGraphic(null);
                } else {
                    field.setText(item);
                    setGraphic(field);
                }
            }
        });
        strength.setMinWidth(72);
        strength.setPrefWidth(72);
        strength.setMaxWidth(90);

        TableColumn<Row, String> result = new TableColumn<>("Result");
        result.setCellValueFactory(c -> new ReadOnlyStringWrapper(
                enabled.isSelected() ? c.getValue().result() : "Chunky default"));
        result.setPrefWidth(110);
        result.setMinWidth(60);

        table.getColumns().setAll(Arrays.asList(name, map, mode, strength, result));
    }

    // ---- packs ----

    private Window window() {
        return box.getScene() != null ? box.getScene().getWindow() : null;
    }

    private void addZipPacks() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Add emission packs");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Resource packs", "*.zip", "*.jar"));
        List<File> files = chooser.showOpenMultipleDialog(window());
        if (files != null) {
            for (File f : files) if (!packs.contains(f)) packs.add(f);
            savePacks();
        }
    }

    private void addFolderPack() {
        DirectoryChooser chooser = new DirectoryChooser();
        chooser.setTitle("Add an unzipped emission pack");
        File dir = chooser.showDialog(window());
        if (dir != null && !packs.contains(dir)) {
            packs.add(dir);
            savePacks();
        }
    }

    private void movePack(int delta) {
        int i = packList.getSelectionModel().getSelectedIndex();
        int j = i + delta;
        if (i < 0 || j < 0 || j >= packs.size()) return;
        File f = packs.remove(i);
        packs.add(j, f);
        packList.getSelectionModel().select(j);
        savePacks();
    }

    private void savePacks() {
        EmissiveSettings.setExtraPacks(new ArrayList<>(packs));
        loadMasks(false, true);
    }

    /**
     * Read (or convert) the packs' emission maps off the UI thread, then refresh the
     * block list, and the render if the maps may have changed.
     */
    private void loadMasks(boolean rescan, boolean refreshRender) {
        List<File> packFiles = EmissiveSettings.packs();
        packStatus.setText(packFiles.isEmpty() ? "No packs." : "Reading emission maps…");
        worker.submit(() -> {
            long start = System.nanoTime();
            if (rescan) {
                EmissivePacks.clearCache();
                EmissiveLibrary.invalidate();
            }
            Map<String, EmissionMask> loaded;
            try {
                loaded = EmissiveLibrary.masks(packFiles);
            } catch (RuntimeException ex) {
                Log.warn("ChunkyCL: could not read emission maps", ex);
                loaded = Collections.emptyMap();
            }
            long ms = (System.nanoTime() - start) / 1_000_000;
            Map<String, EmissionMask> result = loaded;
            Platform.runLater(() -> {
                masks = result;
                masksLoaded = true;
                packStatus.setText(packFiles.isEmpty()
                        ? "No packs. Add a LabPBR or OptiFine emissive pack to use emission maps."
                        : String.format("%d emission maps from %d pack%s (%d ms).",
                                result.size(), packFiles.size(), packFiles.size() == 1 ? "" : "s", ms));
                rebuildRows();
                if (refreshRender) refreshScene();
            });
        });
    }

    // ---- blocks ----

    private void syncFromScene(boolean force) {
        JsonValue settings = scene.getAdditionalData(EmissiveSettings.SCENE_KEY);
        List<Block> palette = scene.getPalette() != null ? scene.getPalette().getPalette() : null;
        int size = palette != null ? palette.size() : -1;
        boolean changed = force || settings != shownSettings || palette != shownPalette || size != shownPaletteSize;
        if (!changed) return;
        updating = true;
        enabled.setSelected(EmissiveSettings.get(scene).enabled);
        updating = false;
        if (masksLoaded || force) rebuildRows();
    }

    private void rebuildRows() {
        List<Block> palette = scene.getPalette() != null ? scene.getPalette().getPalette() : Collections.emptyList();
        shownPalette = palette;
        shownPaletteSize = palette.size();
        shownSettings = scene.getAdditionalData(EmissiveSettings.SCENE_KEY);

        EmissionResolver resolver = EmissionResolver.forMasks(masks);
        EmissiveSettings.SceneSettings settings = EmissiveSettings.get(scene);
        Map<String, boolean[]> hasMap = new TreeMap<>();
        Map<String, Float> emittance = new HashMap<>();
        List<Block> blocks;
        try {
            blocks = new ArrayList<>(palette);
        } catch (RuntimeException ex) {
            blocks = Collections.emptyList();  // palette growing during a chunk load; onChunksLoaded retries
        }
        for (Block b : blocks) {
            if (b == null || b.invisible || EmissionResolver.texturesOf(b).isEmpty()) continue;
            boolean[] m = hasMap.computeIfAbsent(b.name, n -> new boolean[1]);
            try {
                m[0] |= resolver.hasMap(b);
            } catch (RuntimeException ex) {
                // A block whose model cannot be inspected is listed without a map.
            }
            emittance.merge(b.name, b.emittance, Math::max);
        }
        List<Row> list = new ArrayList<>();
        for (Map.Entry<String, boolean[]> e : hasMap.entrySet()) {
            list.add(new Row(e.getKey(), e.getValue()[0], emittance.get(e.getKey()), settings.get(e.getKey())));
        }
        rows.setAll(list);
        updateFilter();
    }

    private void updateFilter() {
        String text = filter.getText() == null ? "" : filter.getText().trim().toLowerCase(Locale.ROOT);
        boolean relevant = onlyRelevant.isSelected();
        filteredRows.setPredicate(r -> (text.isEmpty() || r.name.toLowerCase(Locale.ROOT).contains(text))
                && (!relevant || r.hasMap || r.chunkyEmittance > 0 || r.mode != Mode.DEFAULT));
    }

    private void commit(Map<String, BlockSetting> changes) {
        if (changes.isEmpty()) return;
        EmissiveSettings.setBlocks(scene, changes);
        shownSettings = scene.getAdditionalData(EmissiveSettings.SCENE_KEY);
        table.refresh();
        refreshScene();
    }

    /**
     * Restart the render with a material reload, which makes the GPU renderer repack
     * materials and textures. Scene.refresh(ResetReason) is private.
     */
    private void refreshScene() {
        try {
            Method m = Scene.class.getDeclaredMethod("refresh", ResetReason.class);
            m.setAccessible(true);
            m.invoke(scene, ResetReason.MATERIALS_CHANGED);
        } catch (ReflectiveOperationException | RuntimeException e) {
            Log.warn("ChunkyCL: could not request a material reload; change a material to apply", e);
            scene.refresh();
        }
    }

    @Override
    public void update(Scene scene) {
        boolean newScene = scene != this.scene;
        this.scene = scene;
        syncFromScene(newScene);
    }

    @Override
    public void onChunksLoaded() {
        rebuildRows();
    }

    @Override
    public String getTabTitle() {
        return "Emissive";
    }

    @Override
    public Node getTabContent() {
        return box;
    }
}
