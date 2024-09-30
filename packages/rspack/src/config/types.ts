//#region Resolve
/**
 * Path alias
 * @example
 * ```js
 * {
 * 	"@": path.resolve(__dirname, './src'),
 * 	"abc$": path.resolve(__dirname, './node_modules/abc/index.js'),
 * }
 * // - require("@/a") will attempt to resolve <root>/src/a.
 * // - require("abc") will attempt to resolve <root>/src/abc.
 * // - require("abc/file.js") will not match, and it will attempt to resolve node_modules/abc/file.js.
 * ```
 * */
export type ResolveAlias = {
	[x: string]: string | false | (string | false)[];
};

/** The replacement of [tsconfig-paths-webpack-plugin](https://www.npmjs.com/package/tsconfig-paths-webpack-plugin) in Rspack. */
export type ResolveTsConfig =
	| string
	| {
			configFile: string;
			references?: string[] | "auto" | undefined;
	  };

/** Used to configure the Rspack module resolution */
export type ResolveOptions = {
	/** Path alias */
	alias?: ResolveAlias;

	/** Same as node's [conditionNames](https://nodejs.org/api/packages.html#conditional-exports) for the exports and imports fields in package.json. */
	conditionNames?: string[];

	/**
	 * Parse modules in order.
	 * @default [".js", ".json", ".wasm"]
	 * */
	extensions?: string[];

	/** Redirect module requests when normal resolving fails. */
	fallback?: ResolveAlias;

	/** Try to parse the fields in package.json */
	mainFields?: string[];

	/**
	 * The filename suffix when resolving directories, e.g. require('. /dir/') will try to resolve '. /dir/index'.
	 * @default ['index']
	 */
	mainFiles?: string[];

	/**
	 * The name of the directory to use when resolving dependencies.
	 * @default ["node_modules"]
	 */
	modules?: string[];

	/**
	 * When enabled, require('file') will first look for the . /file file in the current directory, not <modules>/file.
	 * @default false
	 */
	preferRelative?: boolean;

	/**
	 * Opt for absolute paths when resolving, in relation to resolve.roots.
	 * @default false
	 */
	preferAbsolute?: boolean;

	/**
	 * Whether to resolve symlinks to their symlinked location.
	 * @default true
	 */
	symlinks?: boolean;

	/**
	 * By default, It changes to true if resolve.extensions contains an empty string;
	 * otherwise, this value changes to false.
	 */
	enforceExtension?: boolean;

	/**
	 * Customize the imports field in package.json which are used to provide the internal requests of a package (requests starting with # are considered internal).
	 * @default ["imports"]
	 */
	importsFields?: string[];

	/**
	 * The JSON files to use for descriptions.
	 * @default ['package.json']
	 */
	descriptionFiles?: string[];

	/** The replacement of [tsconfig-paths-webpack-plugin](https://www.npmjs.com/package/tsconfig-paths-webpack-plugin) in Rspack. */
	tsConfig?: ResolveTsConfig;

	/**
	 * No longer resolve extensions, no longer resolve mainFiles in package.json (but does not affect requests from mainFiles, browser, alias).
	 * @default false
	 * */
	fullySpecified?: boolean;

	/**
	 * Customize the exports field in package.json.
	 * @default ["exports"]
	 * */
	exportsFields?: string[];

	/** Define alias for the extension. */
	extensionAlias?: Record<string, string | string[]>;

	/**
	 * Define a field, such as browser, that should be parsed in accordance with this [specification](https://github.com/defunctzombie/package-browser-field-spec).
	 * @default ['browser']
	 * */
	aliasFields?: string[];

	/**
	 * A list of resolve restrictions to restrict the paths that a request can be resolved on.
	 * @default []
	 * */
	restrictions?: string[];

	/**
	 * A list of directories where server-relative URLs (beginning with '/') are resolved.
	 * It defaults to the context configuration option.
	 * On systems other than Windows, these requests are initially resolved as an absolute path.
	 * @default []
	 */
	roots?: string[];

	/** Customize the Resolve configuration based on the module type. */
	byDependency?: Record<string, ResolveOptions>;
};

/** Used to configure the Rspack module resolution */
export type Resolve = ResolveOptions;
//#endregion